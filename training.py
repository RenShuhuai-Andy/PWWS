import os
from data_helper.data_helper import DataHelper
from neural_networks import word_cnn, char_cnn, bd_lstm, lstm
import keras
from keras import backend as K
import tensorflow as tf
import argparse
from config import config
from sklearn.utils import shuffle

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
K.set_session(tf.Session(config=tf_config))

parser = argparse.ArgumentParser(
    description='Train a text classifier.')
parser.add_argument('-m', '--model',
                    help='The model of text classifier',
                    choices=['word_cnn', 'char_cnn', 'word_lstm', 'word_bdlstm'],
                    default='word_cnn')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='imdb')
parser.add_argument('-l', '--level',
                    help='The level of process dataset',
                    choices=['word', 'char'],
                    default='word')


def train_text_classifier():
    dataset = args.dataset
    data_helper = DataHelper(dataset, args.level)
    x_train = y_train = x_test = y_test = data_helper.processing()
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    # Take a look at the shapes
    print('dataset: {}; model: {}; level: {}.'.format(dataset, args.model, args.level))
    print('x_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('x_test:', x_test.shape)
    print('y_test:', y_test.shape)

    log_dir = r'./logs/{}/{}/'.format(dataset, args.model)
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True)

    model_filename = r'./runs/{}/{}.dat'.format(dataset, args.model)
    model = batch_size = epochs = None
    assert args.model[:4] == args.level

    if args.model == "word_cnn":
        model = word_cnn(dataset)
        batch_size = config.wordCNN_batch_size[dataset]
        epochs = config.wordCNN_epochs[dataset]
    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
        batch_size = config.bdLSTM_batch_size[dataset]
        epochs = config.bdLSTM_epochs[dataset]
    elif args.model == "char_cnn":
        model = char_cnn(dataset)
        batch_size = config.charCNN_batch_size[dataset]
        epochs = config.charCNN_epochs[dataset]
    elif args.model == "word_lstm":
        model = lstm(dataset)
        batch_size = config.LSTM_batch_size[dataset]
        epochs = config.LSTM_epochs[dataset]

    print('Train...')
    print('batch_size: ', batch_size, "; epochs: ", epochs)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tb_callback])
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    print('Saving model weights...')
    model.save_weights(model_filename)


if __name__ == '__main__':
    args = parser.parse_args()
    train_text_classifier()
