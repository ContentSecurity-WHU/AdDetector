from functools import partial

import torch
from torch.utils.data import DataLoader

import settings
from logger import Logger
from dataset import Dataset
from model import Model
from utils import sentence2tensor, num2one_hot, get_accuracy


if __name__ == '__main__':

    logger = Logger('train')
    logger.info(f'using device: {settings.device}')

    # initialize model
    model = Model(
        vocab_size=settings.Model.vocab_size,
        embedding_size=settings.Model.embedding_size,
        content_size=settings.Model.content_size,
    )

    # prepare training set
    training_ds = Dataset(
        text_path=settings.TrainingPath.texts,
        label_path=settings.TrainingPath.labels,
        transform=partial(
            sentence2tensor,
            content_size=settings.Model.content_size,
            stop_words=None
        ),
        target_transform=partial(
            num2one_hot,
            size=2
        )
    )
    training_dl = DataLoader(training_ds, batch_size=settings.Training.batch_size, shuffle=True)

    # prepare test data
    test_ds = Dataset(
        text_path=settings.TestPath.texts,
        label_path=settings.TestPath.labels,
        transform=partial(
            sentence2tensor,
            content_size=settings.Model.content_size,
            stop_words=None
        )
    )
    test_dl = DataLoader(test_ds, batch_size=len(test_ds))

    # train the model
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=settings.Training.learning_rate)
    for epoch in range(settings.Training.epochs):
        total_loss = 0
        for x, target in training_dl:
            y = model(x)
            target = torch.argmax(target, dim=1)  # ont hot -> argument max
            loss = loss_func(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        logger.debug(
            f'epoch: {epoch + 1}\n'
            f'running loss: {total_loss / len(training_dl)}\n'
        )

    # estimate the model
    x, target = next(iter(test_dl))
    with torch.no_grad():
        y = model(x)
        loss = loss_func(y, target)
        y = torch.argmax(y, dim=1)
        logger.info(
            f'\n\n{"***ESTIMATION***": ^20}\n'
            f'Precision: {get_accuracy(y, target)}\n'
            f'Loss: {loss}'
        )
