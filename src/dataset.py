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
        self.text_path = text_path  # texts may be too large for the memory, so just note their path.
        self.labels = list()
        with open(label_path, 'r') as f:
            for i in f.readlines():
                self.labels.append(int(i))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        text = None
        with open(self.text_path, 'r',encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    text = line
        label = self.labels[idx]

        if self.transform:
            text = self.transform(text)
        if self.target_transform:
            label = self.target_transform(label)

        return text, label


if __name__ == '__main__':
    import settings
    dataset = Dataset(
        text_path=settings.Path.text,
        label_path=settings.Path.label
    )
    print(dataset[0])
