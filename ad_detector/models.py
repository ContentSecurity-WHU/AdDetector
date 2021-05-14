import torch
import torch.nn as nn
import torch.nn.functional as F


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


class TextCNN(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_size: int,
            hidden_size: int,
            filter_size: int,
            dropout: float
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_size, hidden_size, k) for k in filter_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * len(filter_size), 2)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze()
        return x

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.dropout(emb)
        emb = emb.transpose(1, 2)
        out = torch.cat([self.conv_and_pool(emb, conv) for conv in self.convs], -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=-1)
        return out
