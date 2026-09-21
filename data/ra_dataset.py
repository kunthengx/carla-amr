import torch
from torch.utils.data import Dataset


class SaveAugmentedDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            'ts_org': self.data[index],
            'target': self.targets[index],
        }