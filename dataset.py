from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize

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
            image = (image > self.data.at[index, 'Threshold']) * 1
            # Resize
            image = cv2.resize(np.array(image, dtype=np.float32), (1024, 128),
                               interpolation=cv2.INTER_AREA)
            # Rotate
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            transforms = Compose([
                ToTensor(),
            ])
            image = transforms(image)

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
            # form = os.path.join(self.root_dir, 'xml', xml_file)

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
