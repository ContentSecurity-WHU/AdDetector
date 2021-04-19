from typing import List

import jieba
from torch import tensor

from logger import Logger
from settings import device

jieba.setLogLevel('INFO')
word2idx = dict()
idx_cnt = 1  # 词汇编码从1开始，0用于padding


def sentence2tensor(sentence: str, content_size: int, stop_words: List[str] = None) -> tensor:
    global idx_cnt
    words = jieba.lcut(sentence)  # 分词
    if stop_words:
        words = [i for i in words if i not in stop_words]  # 删除停用词
    ret = list()
    for i in words:  # word -> idx
        if i not in word2idx.keys():  # 未收录则添加新词
            word2idx[i] = idx_cnt
            idx_cnt += 1
        ret.append(word2idx[i])
    if len(ret) > content_size:
        Logger('sentence2tensor').warning('content length out of size, result will be truncated.')
    while len(ret) < content_size:  # padding
        ret.append(0)
    ret = ret[:content_size]
    return tensor(ret).to(device)


def num2one_hot(num: int, size: int) -> tensor:
    ret = [0 for _ in range(size)]
    ret[num] = 1
    return tensor(ret).to(device)


if __name__ == '__main__':
    t = sentence2tensor('我爱北京天安门', 10, ['你好', '北京'])
    print(t)
    print(num2one_hot(5, 7))
