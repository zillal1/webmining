from torch.utils.data import Dataset
import torch


class ReviewSet(Dataset):
    def __init__(self, texts, item_func):
        self.texts = texts
        self.item_func = item_func

    def __len__(self): return len(self.texts)

    def __getitem__(self, i): return self.item_func(self.texts[i])


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, item_func):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels
        self.item_func = item_func

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        return self.item_func(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)
