import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm

from torchvision.transforms import functional as F
import albumentations as A
from skimage import io
import cv2

import os
from xml.etree import ElementTree

import transform


class IAM(Dataset):

    def __init__(self, root_dir, data, charset, transforms=None):

        self.root_dir = root_dir
        self.data = data
        self.charset = charset
        self.transforms = transforms

    def __getitem__(self, index):

        image_name = self.data.at[index, 'Image'] + '.png'
        image = self._read_image(image_name)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        else:
            transforms = A.Compose([
                A.augmentations.geometric.Resize(
                    height=128, width=1024, p=1.0, always_apply=True),
                transform.Rotate(always_apply=True, p=1.0),
                transform.ToTensor(always_apply=True, p=1.0)
            ])
            image = transforms(image=image)['image']

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


class Bentham(Dataset):
    def __init__(self, root_dir, data, charset, transforms=None):

        self.root_dir = root_dir
        self.transforms = transforms
        self.data = data
        self.charset = charset

    def __getitem__(self, index):

        image_name = self.data.at[index, 'Image'] + '.png'
        image, bg_color = self._read_image(image_name)

        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        else:
            transforms = A.Compose([
                A.augmentations.geometric.Resize(
                    height=128, width=1024, p=1.0, always_apply=True),
                transform.Rotate(always_apply=True, p=1.0),
                transform.ToTensor(always_apply=True, p=1.0)
            ])
            image = transforms(image=image)['image']

        target = self.data.at[index, 'Transcription']

        return (image, target)

    def __len__(self):
        return len(self.data)

    def _read_image(self, image_name):

        path = os.path.join(self.root_dir, 'BenthamDatasetR0-GT',
                            'Images', 'Lines', image_name)
        # return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        image = io.imread(path, as_gray=False)
        mask = image[:, :, 3]
        image = image[:, :, :-1]

        # skimage.color.rbg2gray converts to float64, uint8 is needed for deslant
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bg_color = np.bincount(gray[0, :]).argmax()
        return (gray * (mask == 255 * 1)) + ((mask == 0) * bg_color).astype(np.uint8), bg_color


class Encoder:

    def __init__(self, charset=None, dataset='IAM'):
        assert dataset in ('IAM', 'Bentham')
        if charset == None:   # When running inference without initializing dataset class
            if dataset == 'Bentham':
                charset = ' !"#&\'()*+,-./0123456789:;<=>?ABCDEFGHIJKLMNOPQRSTUVWXY[]_abcdefghijklmnopqrstuvwxyz|£§àâèéê⊥'
            else:
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


# Reference: https://shoarora.github.io/2020/02/01/collate_fn.html
class Collate:
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, batch):

        images, transcriptions = zip(*batch)

        images = torch.stack(images, dim=0)

        targets, target_lengths = self.encoder.encode(transcriptions)

        return images, targets, target_lengths, transcriptions


def get_charset(data):

    chars = []

    data.Transcription.apply(lambda x: chars.extend(list(x)))
    chars = ''.join(sorted(set(chars)))

    return chars


def create_df(dataset, root_dir):

    if dataset == 'IAM':

        col_names = ['Image', 'Segmentation', 'Transcription', 'Threshold']
        rows = []

        xml_files = sorted(
            glob.glob(os.path.join(root_dir, 'xml', '*.xml')))

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
                    'Image': line_id,
                    'Segmentation': segmentation,
                    'Transcription': transcription,
                    'Threshold': threshold,
                })

        return pd.DataFrame(rows, columns=col_names)

    elif dataset == 'Bentham':
        col_names = ['Image', 'Transcription', 'Length']
        rows = []

        transcription_path = os.path.join(
            root_dir, 'BenthamDatasetR0-GT', 'Transcriptions')

        files = glob.glob(os.path.join(transcription_path, '*.txt'))
        tk = tqdm(files, desc='Loading dataset')

        for file in tk:
            image = os.path.split(file)[-1].split('.')[0]
            transc = open(file).read().strip()
            if len(transc) < 20 or len(transc) > 100:
                continue
            rows.append({
                'Image': image,
                'Transcription': transc,
                'Length': len(transc)
            })

        return pd.DataFrame(rows, columns=col_names)


def dataset(dataset, root_dir,
            csv_file_path=None,
            default_partition=False,
            partition=(0.7, 0.1, 0.2),
            shuffle=True, seed=42,
            train_transform=None,
            val_transform=None,
            test_transform=None):

    assert dataset in ('IAM', 'Bentham')
    assert isinstance(default_partition, bool)
    assert isinstance(partition, tuple)
    assert isinstance(seed, int)

    if csv_file_path is not None:
        data = pd.read_csv(csv_file_path)
    else:
        data = create_df(dataset, root_dir)
        # To load dataset from csv next time with csv_file_path
        data.to_csv(f"{dataset}_df.csv", index=False)

    if default_partition:
        # titanic[titanic["Pclass"].isin([2, 3])]
        if dataset == 'IAM':
            partition_path = os.path.join(
                root_dir, 'LargeWriterIndependentTextLineRecognitionTask')
            train_images = open(os.path.join(
                partition_path, 'trainset.txt')).read().splitlines()
            val_images = open(os.path.join(partition_path, 'validationset1.txt')).read().splitlines(
            ) + open(os.path.join(partition_path, 'validationset2.txt')).read().splitlines()

            test_images = open(os.path.join(
                partition_path, 'testset.txt')).read().splitlines()

        elif dataset == 'Bentham':
            partition_path = os.path.join(
                root_dir, 'BenthamDatasetR0-GT', 'Partitions')
            train_images = open(os.path.join(
                partition_path, 'TrainLines.lst')).read().splitlines()
            val_images = open(os.path.join(
                partition_path, 'TrainLines.lst')).read().splitlines()
            test_images = open(os.path.join(
                partition_path, 'TrainLines.lst')).read().splitlines()

        train_data = data[data.Image.isin(train_images)]
        val_data = data[data.Image.isin(val_images)]
        test_data = data[data.Image.isin(test_images)]

    else:
        dataset_size = len(data)
        indices = list(range(dataset_size))
        num_partition = len(partition)

        if num_partition < 1 or num_partition > 3:
            print(
                "Invalid partition size. Format: (train_partition, validation_partition, test_partition)")
            return
        if num_partition == 3:
            val_size = int(np.floor(partition[1] * dataset_size))
        else:
            val_size = 0
        test_size = int(np.floor(partition[-1] * dataset_size))
        if num_partition > 1:
            if int(sum(partition)) == 1:
                train_size = dataset_size - (test_size + val_size)
            else:
                train_size = int(np.floor(partition[0] * dataset_size))
        else:
            train_size = 0

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)

        train_indices = indices[: train_size]
        val_indices = indices[train_size: train_size + val_size]
        test_indices = indices[train_size +
                               val_size: train_size + val_size + test_size]

        train_data = data.loc[train_indices].reset_index(drop=True)
        val_data = data.loc[val_indices].reset_index(drop=True)
        test_data = data.loc[test_indices].reset_index(drop=True)

    charset = get_charset(data)

    if dataset == 'IAM':

        train_set = IAM(root_dir=root_dir, data=train_data,
                        charset=charset, transforms=train_transform)
        val_set = IAM(root_dir=root_dir, data=val_data,
                      charset=charset, transforms=val_transform)
        test_set = IAM(root_dir=root_dir, data=test_data,
                       charset=charset, transforms=test_transform)

    elif dataset == 'Bentham':
        train_set = Bentham(root_dir=root_dir, data=train_data,
                            charset=charset, transforms=train_transform)
        val_set = Bentham(root_dir=root_dir, data=val_data,
                          charset=charset, transforms=val_transform)
        test_set = Bentham(root_dir=root_dir, data=test_data,
                           charset=charset, transforms=test_transform)

    if default_partition or num_partition == 3:
        return train_set, val_set, test_set

    elif num_partition == 2:
        return train_set, test_set

    else:
        return test_set
