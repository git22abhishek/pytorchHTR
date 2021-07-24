import torch
import numpy as np


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
