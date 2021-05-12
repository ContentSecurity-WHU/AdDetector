import pathlib
import os


src = pathlib.Path(__file__).absolute().parent.parent
project = src.parent
data = project / 'data'

stop_words = data / 'stop_words.txt'
models = data / 'models'

training_texts = data / 'training' / 'texts.txt'
training_labels = data / 'training' / 'labels.txt'

test_texts = data / 'test' / 'texts.txt'
test_labels = data / 'test' / 'labels.txt'

if not os.path.exists(models):
    os.mkdir(models)
