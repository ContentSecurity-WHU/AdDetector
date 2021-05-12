# import "train" function
from ad_detector.train_model import train
# import model
from ad_detector.models import BiLSTM
# import configuration
from ad_detector import config


if __name__ == '__main__':
    # initialize model
    model = BiLSTM(
        vocab_size=config.model.vocab_size,
        embedding_size=config.model.embedding_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        content_size=config.model.content_size,
    ).to(config.device)

    epochs = [5, 10, 15]

    for i in epochs:
        # change epochs
        config.training.epochs = i
        # calculate loss
        training_loss, test_loss = train(model)
        # display (should be replaced with plotting code)
        print(f'epochs: {i}, training loss: {training_loss}, test loss: {test_loss}')
