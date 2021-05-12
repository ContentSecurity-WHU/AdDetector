from pathlib import Path

import torch

from utils import sentence2tensor
from logger import Logger


class AdDetector:
    def __init__(
            self,
            model_path: Path,
            stop_words_path: Path,
            word2idx_path: Path,
            max_content_size: int
    ):
        self.model = torch.load(model_path)
        self.model.eval()
        with open(stop_words_path, 'r', encoding='utf8') as f:
            self.stop_words = f.read().split('\n')
        self.word2idx_path = word2idx_path
        self.max_content_size = max_content_size
        self.logger = Logger(self.__class__.__name__)

    def is_ad(self, sentence: str) -> bool:
        x = sentence2tensor(sentence, self.max_content_size, self.word2idx_path, self.stop_words).unsqueeze(0)
        self.logger.debug(f'word vector: {x}')
        y = self.model(x)
        self.logger.debug(f'model output: {y}')
        prediction = y.argmax(1)
        return prediction.item() == 1


if __name__ == '__main__':
    import config

    print(AdDetector(
        config.training.model_path,
        config.path.stop_words,
        config.utils.word2idx_path,
        config.model.content_size
    ).is_ad('详情请加微信'))
