import time
import sys
from tqdm import trange, tqdm
import torch

from dataset import IAM
from dataloader import CTCDataLoader
import pandas as pd

# def do_something():
#     time.sleep(1)


# def do_another_something():
#     time.sleep(1)


# for i in trange(10, file=sys.stdout, desc='outer loop'):
#     do_something()

#     for j in trange(100, file=sys.stdout, leave=False, unit_scale=True, desc='inner loop'):
#         do_another_something()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_ROOT_DIR = '/mnt/d/Machine-Learning/Datasets/iamdataset/uncompressed'
NUM_EPOCHS = 200

dataset = IAM(DATASET_ROOT_DIR, csv_file_path='iam_df.csv')

df = dataset.data

print(df.head())


# data_loader = CTCDataLoader(dataset, shuffle=True, seed=42, device=DEVICE)

# train_loader, val_loader, test_loader = data_loader(
#     split=(0.6, 0.2, 0.2), batch_size=(8, 16, 16))

# train_loader, val_loader, test_loader = data_loader(
#     split=(0.6, 0.2, 0.2), batch_size=(8, 16, 16))
