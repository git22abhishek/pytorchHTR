import torch
from tqdm import tqdm, trange
import sys

from dataset import IAM
from dataloader import CTCDataLoader
from model import CRNNModel


def train_fn(model, data_loader, ctc_loss_fn, optimizer):

    model.train()

    avg_batch_loss = 0
    tk = tqdm(data_loader, total=len(data_loader),
              leave=False, file=sys.stdout, unit_scale=True, desc="Training")

    for batch in tk:

        images, targets, target_lengths = batch

        optimizer.zero_grad()

        # Forward pass
        preds = model(images)

        # N = batch_size, T = input_length (lenght of squences, ie, time_stamps), C = num_classes
        N, T, C = preds.shape

        # Apply log softmax to classes
        log_probs = torch.nn.functional.log_softmax(preds, dim=2)

        # Permute to bring sequences at first the axis (T x N x C)
        log_probs = log_probs.permute(1, 0, 2).requires_grad_()

        # All input sequences in the batch are of same length
        input_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.long)

        loss = ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)

        loss.backward()

        optimizer.step()

        avg_batch_loss += loss.item()

    tk.clear()
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

            # N = batch_size, T = input_length (lenght of squences, ie, time_stamps), C = num_classes
            N, T, C = batch_preds.shape

            # Apply log softmax to classes
            log_probs = torch.nn.functional.log_softmax(batch_preds, dim=2)

            # Permute to bring sequences at first the axis (T x N x C)
            log_probs = log_probs.permute(1, 0, 2).detach()

            # All input sequences in the batch are of same length
            input_lengths = torch.full(
                size=(N, ), fill_value=T, dtype=torch.long)

            loss = ctc_loss_fn(log_probs, targets,
                               input_lengths, target_lengths)

        avg_val_loss += loss.item()
        preds.append(batch_preds)

    tk.clear()
    tk.close()

    return preds, avg_val_loss/len(data_loader)


def run_training():

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_ROOT_DIR = '/mnt/d/Machine-Learning/Datasets/iamdataset/uncompressed'
    NUM_EPOCHS = 2
    TRAIN_CHECKPOINT_PATH = 'checkpoints/training_state.pth'
    RESUME_TRAINING = False

    dataset = dataset = IAM(DATASET_ROOT_DIR)

    data_loader = CTCDataLoader(dataset, shuffle=True, seed=42, device=DEVICE)

    train_loader, val_loader, test_loader = data_loader(
        split=(0.006, 0.002, 0.002), batch_size=(1, 1, 1))

    model = CRNNModel(vocab_size=len(dataset.charset), time_steps=100)
    model.to(DEVICE)  # Move model to cuda before constructing optimzer for it

    ctc_loss = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=5, verbose=True,
    )

    for epoch in trange(NUM_EPOCHS, file=sys.stdout, desc='EPOCHS'):

        if RESUME_TRAINING or epoch > 0:
            print('Loading model from last checkpoint:...')

            checkpoint = torch.load(TRAIN_CHECKPOINT_PATH)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])

        train_loss = train_fn(model, train_loader, ctc_loss, optimizer)
        valid_preds, valid_loss = eval_fn(model, val_loader, ctc_loss)

        print(
            f'Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}')

        print('Saving model state...')
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'train_loss': train_fn,
            'valid_loss': valid_loss
        }, TRAIN_CHECKPOINT_PATH)


if __name__ == '__main__':
    run_training()
