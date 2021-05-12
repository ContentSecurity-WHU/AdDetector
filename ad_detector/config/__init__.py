import torch

from . import (
    logger,
    model,
    path,
    training,
    utils
)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
