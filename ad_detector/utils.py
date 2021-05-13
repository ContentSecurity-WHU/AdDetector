from typing import List
from pathlib import Path

import jieba
from torch import tensor

from ad_detector.logger import Logger
from ad_detector.config import device

jieba.setLogLevel('INFO')


def sentence2tensor(
        sentence: str,
        content_size: int,
        word2idx: dict,
        stop_words: List[str] = None
) -> tensor:
    words = jieba.lcut(sentence)  # tokenize
    if stop_words is not None:
        words = [i for i in words if i not in stop_words]  # delete stop words
    ret = list()
    for i in words:  # word -> idx
        if i not in word2idx.keys():
            word2idx[i] = len(word2idx) + 1
        ret.append(word2idx[i])
    # if len(ret) > content_size:
    #     Logger('sentence2tensor').warning('content length out of size, result will be truncated.')
    while len(ret) < content_size:  # padding
        ret.append(0)
    ret = ret[:content_size]
    return tensor(ret, device=device)


def num2one_hot(num: int, size: int) -> tensor:
    ret = [0 for _ in range(size)]
    ret[num] = 1
    return tensor(ret, device=device)


def get_accuracy(predictions: tensor, targets: tensor) -> float:
    assert len(predictions) == len(targets)
    hit_cnt = 0
    for i, j in zip(predictions, targets):
        if i == j:
            hit_cnt += 1
    return hit_cnt / len(predictions)
