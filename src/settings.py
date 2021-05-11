import pathlib
import logging

import torch

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type(torch.cuda.torch.FloatTensor)
else:
    device = 'cpu'


class BasicPath:
    # basic path
    src = pathlib.Path(__file__).absolute().parent
    project = src.parent
    data = project / 'data'


class TrainingPath:
    texts = BasicPath.data / 'training' / 'texts.txt'
    labels = BasicPath.data / 'training' / 'labels.txt'


class TestPath:
    texts = BasicPath.data / 'test' / 'texts.txt'
    labels = BasicPath.data / 'test' / 'labels.txt'


class Logger:
    format = '[%(name)-10s] %(levelname)-8s: %(message)s'
    level = logging.DEBUG


class Model:
    vocab_size = 10000
    embedding_size = 50
    content_size = 1000


class Training:
    batch_size = 8
    learning_rate = 0.1
    epochs = 10
