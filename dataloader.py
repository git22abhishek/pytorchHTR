import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from dataset import IAM


# Reference: https://shoarora.github.io/2020/02/01/collate_fn.html
class Collate:
    def __init__(self, encoder):
        self.encoder = encoder

    def __call__(self, batch):

        images, transcriptions = zip(*batch)

        images = torch.stack(images, dim=0)

        targets, target_lengths = self.encoder.encode(transcriptions)

        return images, targets, target_lengths, transcriptions


class CTCDataLoader:

    def __init__(self, ds, encoder, shuffle=True, seed=42, num_workers=2):
        assert isinstance(ds, IAM)
        assert isinstance(shuffle, bool)
        assert isinstance(seed, int)

        self.ds = ds
        self.shuffle = shuffle
        self.seed = seed
        self.encoder = encoder
        self.num_workers = num_workers

    def __call__(self, default_split=False, split=(0.6, 0.2, 0.2), batch_size=(8, 16, 16)):

        if default_split:
            pass
        else:
            dataset_size = len(self.ds)
            indices = list(range(dataset_size))

            if len(split) < 2 or len(split) > 3:
                print(
                    "Invalid split size. Format: (train_split, validation_split, test_split)")
                return

            if len(split) == 3:
                val_size = int(np.floor(split[1] * dataset_size))
            else:
                val_size = 0

            test_size = int(np.floor(split[-1] * dataset_size))
            if int(sum(split)) == 1:
                train_size = dataset_size - (test_size + val_size)
            else:
                train_size = int(np.floor(split[0] * dataset_size))

            if self.shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(indices)

            train_indices = indices[: train_size]
            val_indices = indices[train_size: train_size + val_size]
            test_indices = indices[train_size +
                                   val_size: train_size + val_size + test_size]

            # Creating PT data samplers and loaders:
            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)
            test_sampler = SubsetRandomSampler(test_indices)

            collater = Collate(self.encoder)

            # Dataloader
            train_loader = DataLoader(self.ds, batch_size=batch_size[0],
                                      sampler=train_sampler, collate_fn=collater, num_workers=self.num_workers)
            val_loader = DataLoader(self.ds, batch_size=batch_size[1],
                                    sampler=val_sampler, collate_fn=collater, num_workers=self.num_workers)
            test_loader = DataLoader(self.ds, batch_size=batch_size[-1],
                                     sampler=test_sampler, collate_fn=collater, num_workers=self.num_workers)

            if len(split) == 3:
                return train_loader, val_loader, test_loader

            return train_loader, test_loader

    # def collate_fn(self, batch):
    #     char_dic = {char: index for index,
    #                 char in enumerate(self.ds.charset, 1)}

    #     images, transcriptions = zip(*batch)

    #     images = torch.stack(images, dim=0)

    #     target_lengths = [len(line) for line in transcriptions]
    #     target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    #     targets = []
    #     for line in transcriptions:
    #         targets.extend([char_dic[char] for char in line])
    #     targets = torch.tensor(targets, dtype=torch.long)

    #     dev = self.device
    #     return images.to(dev), targets.to(dev), target_lengths.to(dev)
