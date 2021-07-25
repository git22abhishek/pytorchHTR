from dataset import dataset, Encoder  # , Collate
from model import CRNNModel
import transform

import torch
# from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# from tqdm import tqdm, trange

# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import cv2
from PIL import Image, ImageOps
from io import BytesIO
import base64
import sys


def base64str_to_numpy_array(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    grey_image = ImageOps.grayscale(Image.open(BytesIO(base64bytes)))
    return np.asarray(grey_image, dtype=np.uint8)


def infer(image: np.ndarray):
    encoder = Encoder(dataset='IAM')
    model = CRNNModel(vocab_size=79, time_steps=100)
    dev = 'cpu'

    device = torch.device(dev)
    checkpoint = torch.load(
        'checkpoints/training_state.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    transforms = A.Compose([
        transform.Deslant(always_apply=True, p=1.0),
        A.augmentations.geometric.Resize(
            height=128, width=1024, p=1.0, always_apply=True),
        transform.Rotate(always_apply=True, p=1.0),
        A.augmentations.transforms.Normalize(
            mean=(119.872), std=(54.866), p=1.0, always_apply=True),
        ToTensorV2(always_apply=True, p=1.0),
    ])

    image = transforms(image=image)['image']

    with torch.no_grad():

        model.to(dev)
        images = torch.unsqueeze(image, dim=0)
        images = images.to(dev)

        preds = model(images)
        preds_decoded = encoder.best_path_decode(
            preds, return_text=True)

    return preds_decoded[0]


if __name__ == '__main__':
    if sys.argv[1]:
        image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        print(infer(image))
