from pathlib import Path
import pickle

import torch

from ad_detector.utils import sentence2tensor
from ad_detector.logger import Logger


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

        with open(word2idx_path, 'rb') as f:
            self.word2idx = pickle.load(f)
        self.max_content_size = max_content_size
        self.logger = Logger(self.__class__.__name__)

    def is_ad(self, sentence: str) -> bool:
        x = sentence2tensor(sentence, self.max_content_size, self.word2idx, self.stop_words).unsqueeze(0)
        # self.logger.debug(f'word vector: {x}')
        y = self.model(x)
        # self.logger.debug(f'model output: {y}')
        prediction = y.argmax(1)
        return prediction.item() == 1



