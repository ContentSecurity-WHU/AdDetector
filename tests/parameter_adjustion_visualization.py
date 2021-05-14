# import "train" function
from ad_detector.train_model import train
# import model
from ad_detector.models import BiLSTM
# import configuration
from ad_detector import config
# import plotting
import matplotlib.pyplot as plt
import numpy as np


def batchsize_plot():
    train_loss_dict = {}
    test_loss_dict = {}
    Epoch_dict = {}
    batch_sizes = [50, 75, 100, 125]

    for i in range(4):
        config.training.batch_sizes = batch_sizes[i]
        train_loss_list, test_loss_list, Epoch = train(model)
        train_loss_dict['{}'.format(i)] = train_loss_list
        test_loss_dict['{}'.format(i)] = test_loss_list
        Epoch_dict['{}'.format(i)] = Epoch

    # visualization
    # train_loss
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(Epoch_dict['0'], train_loss_dict['0'], linewidth=2, color='r', label="4")
    plt.plot(Epoch_dict['1'], train_loss_dict['1'], linewidth=2, color='g', label="8")
    plt.plot(Epoch_dict['2'], train_loss_dict['2'], linewidth=2, color='b', label="16")
    plt.plot(Epoch_dict['3'], train_loss_dict['3'], linewidth=2, color='y', label="32")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("change batch_sizes in train", fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    # test_loss
    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(Epoch_dict['0'], test_loss_dict['0'], linewidth=2, color='r', label="50")
    plt.plot(Epoch_dict['1'], test_loss_dict['1'], linewidth=2, color='g', label="75")
    plt.plot(Epoch_dict['2'], test_loss_dict['2'], linewidth=2, color='b', label="100")
    plt.plot(Epoch_dict['3'], test_loss_dict['3'], linewidth=2, color='y', label="125")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("model = {}, change learning_rate in dev".format(model), fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    plt.legend()
    plt.show()


def hiddensize_plot():
    train_loss_dict = {}
    test_loss_dict = {}
    Epoch_dict = {}
    hidden_sizes = [64, 128, 256]

    for i in range(3):
        config.model.hidden_size = hidden_sizes[i]
        train_loss_list, test_loss_list, Epoch = train(model)
        train_loss_dict['{}'.format(i)] = train_loss_list
        test_loss_dict['{}'.format(i)] = test_loss_list
        Epoch_dict['{}'.format(i)] = Epoch

    # visualization
    # train_loss
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(Epoch_dict['0'], train_loss_dict['0'], linewidth=2, color='r', label="64")
    plt.plot(Epoch_dict['1'], train_loss_dict['1'], linewidth=2, color='g', label="128")
    plt.plot(Epoch_dict['2'], train_loss_dict['2'], linewidth=2, color='b', label="256")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("change hidden_sizes in train", fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    # test_loss
    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(Epoch_dict['0'], test_loss_dict['0'], linewidth=2, color='r', label="64")
    plt.plot(Epoch_dict['1'], test_loss_dict['1'], linewidth=2, color='g', label="128")
    plt.plot(Epoch_dict['2'], test_loss_dict['2'], linewidth=2, color='b', label="256")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("change hidden_sizes in train", fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    plt.legend()
    plt.show()


def embeddingsize_plot():
    train_loss_dict = {}
    test_loss_dict = {}
    Epoch_dict = {}
    embedding_sizes = [64, 128, 256]

    for i in range(3):
        config.model.embedding_sizes = embedding_sizes[i]
        train_loss_list, test_loss_list, Epoch = train(model)
        train_loss_dict['{}'.format(i)] = train_loss_list
        test_loss_dict['{}'.format(i)] = test_loss_list
        Epoch_dict['{}'.format(i)] = Epoch

    # visualization
    # train_loss
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(Epoch_dict['0'], train_loss_dict['0'], linewidth=2, color='r', label="64")
    plt.plot(Epoch_dict['1'], train_loss_dict['1'], linewidth=2, color='g', label="128")
    plt.plot(Epoch_dict['2'], train_loss_dict['2'], linewidth=2, color='b', label="256")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("change embedding_sizes in train", fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    # test_loss
    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(Epoch_dict['0'], test_loss_dict['0'], linewidth=2, color='r', label="64")
    plt.plot(Epoch_dict['1'], test_loss_dict['1'], linewidth=2, color='g', label="128")
    plt.plot(Epoch_dict['2'], test_loss_dict['2'], linewidth=2, color='b', label="256")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("change embedding_sizes in train", fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    plt.legend()
    plt.show()


def dropout_plot():
    train_loss_dict = {}
    test_loss_dict = {}
    Epoch_dict = {}
    dropout = [0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(3):
        config.model.dropout = dropout[i]
        train_loss_list, test_loss_list, Epoch = train(model)
        train_loss_dict['{}'.format(i)] = train_loss_list
        test_loss_dict['{}'.format(i)] = test_loss_list

    # visualization
    # train_loss
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(Epoch_dict['0'], train_loss_dict['0'], linewidth=2, color='r', label="0.1")
    plt.plot(Epoch_dict['1'], train_loss_dict['1'], linewidth=2, color='g', label="0.2")
    plt.plot(Epoch_dict['2'], train_loss_dict['2'], linewidth=2, color='b', label="0.3")
    plt.plot(Epoch_dict['3'], train_loss_dict['3'], linewidth=2, color='y', label="0.4")
    plt.plot(Epoch_dict['4'], train_loss_dict['4'], linewidth=2, color='m', label="0.5")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("change dropout in train", fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    # test_loss
    plt.figure(1)
    plt.subplot(1, 2, 2)
    plt.plot(Epoch_dict['0'], test_loss_dict['0'], linewidth=2, color='r', label="0.1")
    plt.plot(Epoch_dict['1'], test_loss_dict['1'], linewidth=2, color='g', label="0.2")
    plt.plot(Epoch_dict['2'], test_loss_dict['2'], linewidth=2, color='b', label="0.3")
    plt.plot(Epoch_dict['3'], test_loss_dict['3'], linewidth=2, color='y', label="0.4")
    plt.plot(Epoch_dict['4'], test_loss_dict['4'], linewidth=2, color='m', label="0.5")

    plt.ylim((0, 1))
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)  # 设置刻度
    plt.title("change dropout in train", fontsize=10)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("loss", fontsize=14)

    plt.legend()
    plt.show()


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

    #train(model)
    # batchsize_plot()
    # hiddensize_plot()
    embeddingsize_plot()
    # dropout_plot()
