import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
            self,
            vocab_size: int,  # size of vocabulary
            embedding_size: int,  # length of word vector
            hidden_size: int,  # size of LSTM hidden layer
            num_layers: int,  # number of layers in LSTM
            dropout: float,  # rate of dropped neurons in LSTM
            content_size: int,  # length of input vector
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, 0)
        self.lstm = nn.LSTM(
            embedding_size, hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout
        )
        self.flatten = nn.Flatten()
        self.l_r_stack = nn.Sequential(
            nn.Linear(2 * hidden_size * content_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.softmax = nn.Softmax(1)

    def forward(self, x: torch.tensor):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.flatten(x)
        x = self.l_r_stack(x)
        x = self.softmax(x)
        return x
