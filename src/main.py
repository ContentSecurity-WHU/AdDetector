from functools import partial

import torch
from torch.utils.data import DataLoader

import settings
from logger import Logger
from dataset import Dataset
from model import Model
from utils import sentence2tensor, num2one_hot


if __name__ == '__main__':

    logger = Logger('main')
    logger.info(f'using device: {settings.device}')

    model = Model(
        vocab_size=settings.Model.vocab_size,
        embedding_size=settings.Model.embedding_size,
        hidden_size=settings.Model.hidden_size,
        content_size=settings.Model.content_size,
        label_num=settings.Model.label_num
    ).to(settings.device)
    dataset = Dataset(
        text_path=settings.Path.text,
        label_path=settings.Path.label,
        transform=partial(
            sentence2tensor,
            content_size=settings.Model.content_size,
            stop_words=None
        ),
        target_transform=partial(
            num2one_hot,
            size=settings.Model.label_num
        )
    )
    dl = DataLoader(dataset, batch_size=settings.Train.batch_size)

    # 训练前，测试准确率
    total = 0
    hit = 0
    with torch.no_grad():
        for x, target in dl:
            y = model(x)
            for i, j in zip(y, target):
                total += 1
                if torch.argmax(i) == torch.argmax(j):
                    hit += 1
    logger.info(f'accuracy before training {hit / total * 100}%')

    # 训练
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.Train.learning_rate)
    for i in range(settings.Train.epochs):
        epoch = i + 1

        for x, target in dl:
            y = model(x)
            target = torch.argmax(target, dim=1)
            loss = loss_func(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 训练后，再次测试准确率
    total = 0
    hit = 0
    with torch.no_grad():
        for x, target in dl:
            y = model(x)
            for i, j in zip(y, target):
                total += 1
                if torch.argmax(i) == torch.argmax(j):
                    hit += 1
    logger.info(f'accuracy after training {hit / total * 100}%')
