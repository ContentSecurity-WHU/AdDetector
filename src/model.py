import torch
import torch.nn as nn

from logger import Logger

from settings import device


class Model(nn.Module):
    def __init__(
            self,
            vocab_size: int,  # 总词汇量
            embedding_size: int,  # 词向量维度
            hidden_size: int,  # 隐藏层节点数
            content_size: int,  # 输入向量长度
            label_num: int  # 标签种类数
    ):
        super().__init__()
        self.logger = Logger(self.__class__.__name__)
        self.embedding = nn.Embedding(vocab_size, embedding_size, 0)
        self.l_r_stack = nn.Sequential(
            nn.Linear(embedding_size * content_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, label_num),
        )
        self.softmax = nn.Softmax(0)
        self.logger.debug(self)

    def forward(self, x: torch.tensor):
        # N = batch_size
        # C = content_size
        # E = embedding_size
        # H = hidden_size
        # O = label_num
        y = list()
        self.logger.debug(f'x: {x.size()}')
        for i in x:  # (N, C) -> C
            self.logger.debug(f'i: {i.size()}, {i}')
            i = self.embedding(i)  # C -> (C , E)
            self.logger.debug(f'after embedding, i: {i.size()}')
            i = torch.flatten(i)  # (C, E) -> C * E
            self.logger.debug(f'after flatten, i: {i.size()}')
            i = self.l_r_stack(i)  # C * E -> C * E * H -> O
            self.logger.debug(f'after l&r, i: {i.size()}')
            i = self.softmax(i)
            self.logger.debug(f'after softmax, i: {i.size()}')
            y.append(i)  # O -> (N, O)
        y = torch.stack(y).to(device)
        self.logger.debug(f'y: {y.size()}')
        return y
