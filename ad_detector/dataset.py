from pathlib import Path
from typing import Callable

import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(
            self,
            text_path: Path,
            label_path: Path,
            transform: Callable = None,
            target_transform: Callable = None
    ):
        with open(text_path, 'r', encoding='utf8') as f:
            self.texts = f.read().split('\n')
        self.labels = list()
        with open(label_path, 'r') as f:
            for i in f.readlines():
                self.labels.append(int(i))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)

        return text, label
