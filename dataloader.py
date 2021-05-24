import torch

class CollateEncoder:
    def __init__(self, charset):
        self.char_dic = {char:index for index, char in enumerate(charset, 1)}
        
    def __call__(self, batch):
        images, transcriptions = zip(*batch)
        
        images = torch.stack(images, dim=0)
        
        target_lengths = [len(line) for line in transcriptions] 
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)
        targets = []
        for line in transcriptions:
            targets.extend([self.char_dic[char] for char in line])
        targets = torch.tensor(targets, dtype=torch.long)   
        
        return images, targets, target_lengths