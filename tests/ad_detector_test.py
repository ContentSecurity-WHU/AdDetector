from ad_detector import config
from ad_detector.ad_detector import AdDetector

if __name__ == '__main__':

    print(AdDetector(
        config.training.model_path,
        config.path.stop_words,
        config.utils.word2idx_path,
        config.model.content_size
    ).is_ad('限时促销'))
