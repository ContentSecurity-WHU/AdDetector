import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
            self,
            vocab_size: int,  # size of vocabulary
            embedding_size: int,  # length of word vector
            content_size: int,  # length of input vector
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, 0)
        self.flatten = nn.Flatten()
        self.l_r_stack = nn.Sequential(
            nn.Linear(embedding_size * content_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.tensor):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.l_r_stack(x)
        x = self.softmax(x)
        return x
