import pathlib
import logging

import torch


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


class Path:
    src = pathlib.Path(__file__).absolute().parent
    project = src.parent
    data = project / 'data'
    text = data / 'texts.txt'
    label = data / 'labels.txt'


class Logger:
    format = '[%(name)-10s] %(levelname)-8s: %(message)s'
    level = logging.INFO


class Model:
    vocab_size = 1000
    embedding_size = 50
    hidden_size = 128
    content_size = 1000
    label_num = 2


class Train:
    batch_size = 8
    learning_rate = 0.1
    epochs = 2
