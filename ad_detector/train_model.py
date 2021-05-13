import pickle
from functools import partial
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader

from ad_detector import config
from ad_detector.logger import Logger
from ad_detector.dataset import Dataset
from ad_detector.utils import sentence2tensor, num2one_hot, get_accuracy


def train(model) -> Tuple[List, List, List]:
    logger = Logger('train')
    logger.info(f'using device: {config.device}')

    # load word2idx
    try:
        with open(config.path.data / 'word2idx.pkl', 'rb') as f:
            word2idx = pickle.load(f)
    except FileNotFoundError:
        word2idx = dict()
    logger.debug(f'word2idx length: {len(word2idx)}')

    # load stop words
    with open(config.path.stop_words, 'r', encoding='utf8') as f:
        stop_words = f.read().split('\n')

    # prepare training set
    training_ds = Dataset(
        text_path=config.path.training_texts,
        label_path=config.path.training_labels,
        transform=partial(
            sentence2tensor,
            content_size=config.model.content_size,
            word2idx=word2idx,
            stop_words=stop_words
        ),
        target_transform=partial(
            num2one_hot,
            size=2
        )
    )
    training_dl = DataLoader(training_ds, batch_size=config.training.batch_size, shuffle=True)

    # prepare test data
    test_ds = Dataset(
        text_path=config.path.test_texts,
        label_path=config.path.test_labels,
        transform=partial(
            sentence2tensor,
            content_size=config.model.content_size,
            word2idx=word2idx,
            stop_words=stop_words
        )
    )
    test_dl = DataLoader(test_ds, batch_size=config.training.batch_size)

    # train the model
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    test_loss = list()
    training_loss = list()
    epoch_list = list()

    for epoch in range(config.training.epochs):

        epoch_list.append(epoch)
        for batch, (x, target) in enumerate(training_dl):
            y = model(x)
            target = torch.argmax(target, dim=1)  # ont hot -> argument max
            loss = loss_func(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                logger.debug(f'batch:{batch:>5}, loss:{loss:7.10f}')

        # estimate the model
        with torch.no_grad():

            total_loss = 0
            hit = 0
            for x, target in training_dl:
                y = model(x)
                target = torch.argmax(target, dim=1)
                loss = loss_func(y, target)
                y = torch.argmax(y, dim=1)
                total_loss += loss
                for i, j in zip(y, target):
                    if i == j:
                        hit += 1
            training_precision = hit / len(training_ds)
            training_loss.append(total_loss / len(training_dl))

            total_loss = 0
            hit = 0
            for x, target in test_dl:
                y = model(x)
                target = target.to(config.device)
                loss = loss_func(y, target)
                y = torch.argmax(y, dim=1)
                total_loss += loss
                for i, j in zip(y, target):
                    if i == j:
                        hit += 1
            test_precision = hit / len(test_ds)
            test_loss.append(total_loss / len(test_dl))
            logger.info(
                f'\n'
                f'epoch: {epoch + 1}\n'
                f'training loss: {training_loss[-1]}\n'
                f'test loss: {test_loss[-1]}\n'
                f'precision on training set: {training_precision}\n'
                f'precision on test set: {test_precision}\n'
            )

    # save the model
    torch.save(model, config.path.models / 'BiLSTM.model')
    logger.info(f'model is saved in {config.training.model_path}')

    # save word2idx
    with open(config.path.data / 'word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    return training_loss, test_loss, epoch_list
