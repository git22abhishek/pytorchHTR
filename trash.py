import torch
from tqdm import tqdm, trange
import sys
import wandb

from dataset import IAM
from dataloader import CTCDataLoader
from model import CRNNModel
from config import config


def calculate_ctc_loss(ctc_loss_fn, preds, targets, target_lengths):
    # N = batch_size, T = input_length (lenght of squences, ie, time_stamps), C = num_classes
    N, T, C = preds.shape

    # Apply log softmax to classes
    log_probs = torch.nn.functional.log_softmax(preds, dim=2)

    # Permute to bring sequences at first the axis (T x N x C)
    log_probs = log_probs.permute(1, 0, 2).requires_grad_()

    # All input sequences in the batch are of same length
    input_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.long)

    loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

    return loss


def train_log(loss, example_ct, epoch):
    loss = float(loss)

    # where the magic happens
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def train_fn(model, data_loader, ctc_loss_fn, optimizer, epoch):

    model.train()

    avg_batch_loss = 0

    tk = tqdm(data_loader, total=len(data_loader),
              leave=False, file=sys.stdout, unit_scale=True, desc="Training")

    for i, batch in enumerate(tk):

        images, targets, target_lengths = batch

        # Remove gradients from previous update
        optimizer.zero_grad()

        # Forward pass
        preds = model(images)

        # Calculate loss
        loss = calculate_ctc_loss(ctc_loss_fn, preds, targets, target_lengths)

        # Backward pass
        loss.backward()

        # Update Gradients
        optimizer.step()

        # Report metrics every 25th batch
        if ((i + 1) % 25) == 0:
            train_log(loss, images.shape[0], epoch)

        avg_batch_loss += loss.item()

    tk.close()

    return avg_batch_loss/len(data_loader)


def eval_fn(model, data_loader, ctc_loss_fn):

    model.eval()

    avg_val_loss = 0
    tk = tqdm(data_loader, total=len(data_loader),
              leave=False, unit_scale=True, desc="Validating")

    preds = []

    for batch in tk:

        with torch.no_grad():

            images, targets, target_lengths = batch
            batch_preds = model(images)

            loss = calculate_ctc_loss(
                ctc_loss_fn, batch_preds, targets, target_lengths)

        avg_val_loss += loss.item()
        preds.append(batch_preds)

    tk.close()

    return preds, avg_val_loss/len(data_loader)


def training_pipeline(config):

    wandb.login()

    with wandb.init(project="pytorch-demo", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = dataset = IAM(config.data.DATASET_ROOT_DIR,
                                csv_file_path=config.data.CSV_FILE_PATH)

        data_loader = CTCDataLoader(
            ds=dataset,
            shuffle=config.data.SHUFFLE,
            seed=config.data.SEED,
            num_workers=config.data.NUM_WORKERS, device=DEVICE
        )

        train_loader, val_loader, test_loader = data_loader(
            default_split=config.data.DEFAULT_SPLIT,
            split=config.data.SPLIT,
            batch_size=config.data.BATCH_SIZE
        )

        ctc_loss = torch.nn.CTCLoss(
            blank=config.ctc_loss.BLANK,
            reduction=config.ctc_loss.REDUCTION,
            zero_infinity=config.ctc_loss.ZERO_INFINITY
        )

        model = CRNNModel(vocab_size=len(dataset.charset),
                          time_steps=config.TIME_STEPS)
        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), config.opt.LR)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=config.sch.FACTOR,
            patience=config.sch.PATIENCE,
            verbose=config.sch.VERBOSE,
        )

        if config.RESUME_TRAINING:
            print('Loading model from last checkpoint:...')

            checkpoint = torch.load(config.TRAIN_CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])

            tk = tqdm(
                range(checkpoint['epoch'], config.NUM_EPOCHS), file=sys.stdout, desc='EPOCHS')

        else:
            tk = tqdm(range(config.NUM_EPOCHS), file=sys.stdout, desc='EPOCHS')

        # tell wandb to watch what the model gets up to: gradients, weights, and more
        wandb.watch(model, ctc_loss, log="all", log_freq=10)

        for epoch in tk:
            train_loss = train_fn(model, train_loader,
                                  ctc_loss, optimizer, epoch)
            valid_preds, valid_loss = eval_fn(model, val_loader, ctc_loss)

            print(
                f'Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')

            print('Saving model state...')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }, config.TRAIN_CHECKPOINT_PATH)


if __name__ == '__main__':
    training_pipeline(config)
