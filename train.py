import torch
from torch.utils.data import DataLoader
import albumentations as A
from tqdm import tqdm, trange
import sys
import wandb

from dataset import dataset, Encoder, Collate
from model import CRNNModel
from config import config
import transform


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


def train_log(train_loss, val_loss, example_ct, epoch):

    # where the magic happens
    wandb.log({"epoch": epoch, "train_loss": train_loss, "valid_loss": val_loss,
               "num_samples": example_ct}, step=example_ct)
    print(f"Train loss: {train_loss}, Validation loss: {val_loss} after " +
          str(example_ct).zfill(5) + f" examples.")


def train_fn(model, ctc_loss_fn, optimizer, scheduler, dev, train_loader, val_loader, tk, ck_path):

    # tell wandb to watch what the model gets up to: gradients, weights, and more
    wandb.watch(model, ctc_loss_fn, log="all", log_freq=10)

    # Run training and track with wandb
    example_ct = 0  # number of examples (images) seen

    for epoch in tk:

        train_tk = tqdm(train_loader, total=len(train_loader),
                        leave=False, file=sys.stdout, unit_scale=True, desc="Training")

        avg_epoch_loss = 0

        for i, batch in enumerate(train_tk):

            model.train()

            images, targets, target_lengths, _ = batch
            example_ct += images.shape[0]

            # Move to GPU if available
            model.to(dev)
            images = images.to(dev)
            targets = targets.to(dev)
            target_lengths = target_lengths.to(dev)

            # Remove gradients from previous update
            optimizer.zero_grad()

            # Forward pass
            preds = model(images)

            # Calculate loss
            loss = calculate_ctc_loss(
                ctc_loss_fn, preds, targets, target_lengths)
            train_loss = loss.item()
            avg_epoch_loss += train_loss

            # Backward pass
            loss.backward()

            # Update Gradients
            optimizer.step()

            # Validate and report metrics every 25th batch
            if ((i + 1) % 25) == 0:
                print("Validating...")

                model.eval()

                avg_val_loss = 0

                for batch in val_loader:
                    images, targets, target_lengths, _ = batch

                    with torch.no_grad():

                        images, targets, target_lengths, targets_original = batch

                        # Move to GPU if available
                        model.to(dev)
                        images = images.to(dev)
                        targets = targets.to(dev)
                        target_lengths = target_lengths.to(dev)

                        preds = model(images)

                        val_loss = calculate_ctc_loss(
                            ctc_loss_fn, preds, targets, target_lengths)
                        avg_val_loss += val_loss.item()

                avg_val_loss = avg_val_loss / len(val_loader)

                train_log(train_loss, avg_val_loss, example_ct, epoch)

        avg_epoch_loss = avg_epoch_loss / len(train_loader)
        print(
            f'Epoch: {epoch}, Avg Train Loss: {avg_epoch_loss}')

        print('Saving model state...')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_loss': train_loss,
        }, ck_path)

    return model


def eval_fn(model, data_loader, ctc_loss_fn, dev, encoder, epoch):

    model.eval()

    avg_val_loss = 0
    tk = tqdm(data_loader, total=len(data_loader),
              leave=False, unit_scale=True, desc="Validating")

    preds = []

    for batch in tk:

        with torch.no_grad():

            images, targets, target_lengths, targets_original = batch

            # Move to GPU if available
            model.to(dev)
            images = images.to(dev)
            targets = targets.to(dev)
            target_lengths = target_lengths.to(dev)

            batch_preds = model(images)

            loss = calculate_ctc_loss(
                ctc_loss_fn, batch_preds, targets, target_lengths)

        avg_val_loss += loss.item()
        preds.append(batch_preds)

    tk.close()

    avg_val_loss = avg_val_loss / len(data_loader)

    wandb.log({'avg_val_loss': avg_val_loss})

    # trans = []
    # for pred in preds:
    #     transcription = encoder.best_path_decode(pred, return_text=True)
    #     trans.append((trans, targets_original))

    return preds, avg_val_loss


def training_pipeline(config):

    wandb.login()

    with wandb.init(project="handwriting-recognition", config=config):
        # # access all HPs through wandb.config, so logging matches execution!
        # config = wandb.config

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        train_transforms = A.Compose([
            transform.Deslant(always_apply=True, p=1.0),
            transform.Binarize(p=0.3),
            A.augmentations.geometric.transforms.Affine(
                scale=0.8, shear=(-3, 3), cval=(195, 255), p=0.8,
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}),
            A.augmentations.geometric.Resize(
                height=128, width=1024, p=1.0, always_apply=True),
            A.augmentations.transforms.Blur(blur_limit=(3, 4), p=0.4),
            A.augmentations.transforms.GaussNoise(var_limit=(
                10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.3),
            transform.Binarize(p=0.3),
            transform.Rotate(always_apply=True, p=1.0),
            transform.ToTensor(always_apply=True, p=1.0)
        ])

        trainset, validset, testset = dataset(
            'IAM', config.data.DATASET_ROOT_DIR,
            csv_file_path=config.data.CSV_FILE_PATH,
            default_partition=config.data.DEFAULT_PARTITION,
            partition=config.data.PARTITION,
            shuffle=config.data.SHUFFLE,
            seed=config.data.SEED,
            train_transform=train_transforms
        )

        encoder = Encoder(trainset.charset)
        collater = Collate(encoder)

        train_loader = DataLoader(trainset,
                                  batch_size=config.data.BATCH_SIZE[0],
                                  shuffle=config.data.SHUFFLE,
                                  collate_fn=collater,
                                  num_workers=config.data.NUM_WORKERS
                                  )

        val_loader = DataLoader(validset,
                                batch_size=config.data.BATCH_SIZE[1],
                                shuffle=config.data.SHUFFLE,
                                collate_fn=collater, num_workers=config.data.NUM_WORKERS
                                )
        test_loader = DataLoader(testset,
                                 batch_size=config.data.BATCH_SIZE[-1],
                                 shuffle=False, collate_fn=collater,
                                 num_workers=config.data.NUM_WORKERS
                                 )

        ctc_loss = torch.nn.CTCLoss(
            blank=config.ctc_loss.BLANK,
            reduction=config.ctc_loss.REDUCTION,
            zero_infinity=config.ctc_loss.ZERO_INFINITY
        )

        model = CRNNModel(vocab_size=len(trainset.charset),
                          time_steps=config.TIME_STEPS)

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

        model = train_fn(model, ctc_loss, optimizer, scheduler, DEVICE,
                         train_loader, val_loader, tk, config.TRAIN_CHECKPOINT_PATH)

        return model, test_loader


if __name__ == '__main__':
    training_pipeline(config)
