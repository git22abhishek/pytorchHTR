import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import albumentations as A

import pandas as pd
import numpy as np
import glob
import cv2
from tqdm import tqdm

from xml.etree import ElementTree
import os

from deslant import deslant


class IAM(Dataset):

    def __init__(self, root_dir, csv_file_path=None, transforms=None):

        self.root_dir = root_dir
        self.transforms = transforms

        if csv_file_path is not None:
            self.data = pd.read_csv(csv_file_path)
        else:
            self.data = self._create_df()

        self.charset = self.get_charset()

    def __getitem__(self, index):

        image_name = self.data.at[index, 'Image']
        image = self._read_image(image_name)

        if self.transforms is not None:
            image = self.transforms(image)
        else:
            # Deslant
            image = deslant(image, bg_color=255).img
            # Binarize
            # image = (image > int(self.data.at[index, 'Threshold'])) * 1
            _, image = cv2.threshold(
                image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            transforms = A.Compose([
                A.augmentations.geometric.transforms.Affine(
                    fit_output=True, shear=(-5, 5), cval=1, p=1.0),
                A.augmentations.geometric.Resize(
                    height=128, width=1024, always_apply=True),
                # A.augmentations.transforms.Blur(blur_limit=(3, 4), p=0.4),
            ])
            image = transforms(image=image)['image']

            # Rotate
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # Convert from uint8 to float and bring channel to first axis
            image = F.to_tensor(image)

        target = self.data.at[index, 'Transcription']

        return (image, target)

    def __len__(self):
        return len(self.data)

    def _read_image(self, image_name):

        path = image_name.split('-')  # ['a01', '000u', '00.png']
        path = os.path.join(
            self.root_dir,
            'lines',
            path[0],
            '-'.join(path[:2]),
            image_name
        )

        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def _create_df(self):

        col_names = ['Image', 'Segmentation', 'Transcription', 'Threshold']
        rows = []

        xml_files = sorted(
            glob.glob(os.path.join(self.root_dir, 'xml', '*.xml')))

        tk = tqdm(xml_files, desc='Loading dataset')

        for xml_file in tk:
            # Parse the xml file
            dom = ElementTree.parse(xml_file)
            root = dom.getroot()

            # Iterate through all the lines in a form
            for line in root.iter('line'):

                transcription = line.attrib['text'].replace('&quot;', '"')
                # result of segmentation, either 'ok' or 'err'
                segmentation = line.attrib['segmentation']
                line_id = line.attrib['id']
                # threshold for binarization
                threshold = line.attrib['threshold']

                rows.append({
                    'Image': line_id + '.png',
                    'Segmentation': segmentation,
                    'Transcription': transcription,
                    'Threshold': threshold,
                })

        return pd.DataFrame(rows, columns=col_names)

    def get_charset(self):

        data = self.data
        chars = []

        data.Transcription.apply(lambda x: chars.extend(list(x)))
        chars = ''.join(sorted(set(chars)))

        return chars


class Encoder:

    def __init__(self, charset=None):
        if charset == None:  # When running inference without initializing dataset class
            # IAM character set
            charset = ' !"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self.charset = charset

    def encode(self, transcriptions):
        char_dic = {char: index for index,
                    char in enumerate(self.charset, 1)}

        target_lengths = [len(line) for line in transcriptions]
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        targets = []
        for line in transcriptions:
            targets.extend([char_dic[char] for char in line])
        targets = torch.tensor(targets, dtype=torch.long)

        return targets, target_lengths

    def best_path_decode(self, predictions, return_text=False):

        softmax_out = predictions.softmax(2).argmax(2).detach().cpu().numpy()

        decoded = []
        for i in range(0, softmax_out.shape[0]):
            dup_rm = softmax_out[i, :][np.insert(
                np.diff(softmax_out[i, :]).astype(bool), 0, True)]
            dup_rm = dup_rm[dup_rm != 0]
            decoded.append(dup_rm.astype(int))

        if not return_text:
            return decoded

        transcriptions = []
        for line in decoded:
            pred = ''.join([self.charset[letter-1] for letter in line])
            transcriptions.append(pred)

        return transcriptions
