from functools import partial

import torch
from torch.utils.data import DataLoader

import config
from logger import Logger
from dataset import Dataset
from models import BiLSTM
from utils import sentence2tensor, num2one_hot, get_accuracy


if __name__ == '__main__':

    logger = Logger('train')
    logger.info(f'using device: {config.device}')

    # initialize model
    model = BiLSTM(
        vocab_size=config.model.vocab_size,
        embedding_size=config.model.embedding_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        content_size=config.model.content_size,
    ).to(config.device)

    # read stop words
    with open(config.path.stop_words, 'r', encoding='utf8') as f:
        stop_words = f.read().split('\n')

    # prepare training set
    training_ds = Dataset(
        text_path=config.path.training_texts,
        label_path=config.path.test_labels,
        transform=partial(
            sentence2tensor,
            content_size=config.model.content_size,
            dict_path=config.utils.word2idx_path,
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
            dict_path=config.utils.word2idx_path,
            stop_words=stop_words
        )
    )
    test_dl = DataLoader(test_ds, batch_size=len(test_ds))

    # train the model
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate)
    for epoch in range(config.training.epochs):
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
        target = target.to(config.device)
        loss = loss_func(y, target)
        y = torch.argmax(y, dim=1)
        logger.info(
            f'\n\n{"***ESTIMATION***": ^20}\n'
            f'Precision: {get_accuracy(y, target)}\n'
            f'Loss: {loss}\n'
        )

    # save the model
    torch.save(model, config.path.models / 'BiLSTM.model')
    logger.info(f'model is saved in {config.training.model_path}')
