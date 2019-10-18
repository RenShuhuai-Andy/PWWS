import os
import numpy as np
from data_helper.data_helper import DataHelper
from data_helper.word_level_process import get_tokenizer
from neural_networks import word_cnn, char_cnn, bd_lstm, lstm
from evaluate_fool_results import process_adversarial_data
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
parser.add_argument('--clean_samples_cap',
                    help='Amount of clean(test) samples to fool',
                    type=int, default=1000)
parser.add_argument('-m', '--model',
                    help='The model of text classifier',
                    choices=['word_cnn', 'char_cnn', 'word_lstm', 'word_bdlstm'],
                    default='word_cnn')
parser.add_argument('-d', '--dataset',
                    help='Data set',
                    choices=['imdb', 'agnews', 'yahoo'],
                    default='yahoo')
parser.add_argument('-l', '--level',
                    help='The level of process dataset',
                    choices=['word', 'char'],
                    default='word')


def adversarial_training():
    clean_samples_cap = args.clean_samples_cap  # 1000

    # Load and process clean data
    dataset = args.dataset
    tokenizer = get_tokenizer(dataset)
    data_helper = DataHelper(dataset, args.level)
    x_train, y_train, x_test, y_test = data_helper.processing()

    # Take a look at the shapes
    print('dataset: {}; model: {}; level: {}.'.format(dataset, args.model, args.level))
    print('x_train:', x_train.shape)
    print('y_train:', y_train.shape)
    print('x_test:', x_test.shape)
    print('y_test:', y_test.shape)

    # Load and process adv data
    adv_text_filename = r'./fool_result/{}/{}/adv_{}.txt'.format(dataset, args.model, str(clean_samples_cap))
    print('adversarial file:', adv_text_filename)
    x_adv = process_adversarial_data(adv_text_filename, args.level, dataset, tokenizer)
    print('x_adv:', x_adv.shape)

    # Add adv data to traing set and shuffle
    x_train_plus = np.vstack((x_train, x_adv))
    y_train_plus = np.vstack((y_train, y_test[:x_adv.shape[0]]))
    x_train_plus, y_train_plus = shuffle(x_train_plus, y_train_plus)
    print('x_train_plus:', x_train_plus.shape)
    print('y_train_plus:', y_train_plus.shape)

    adv_log_path = r'./logs/{}/adv_{}/'.format(dataset, args.model)
    if not os.path.exists(adv_log_path):
        os.makedirs(adv_log_path)
    tb_callback = keras.callbacks.TensorBoard(log_dir=adv_log_path, histogram_freq=0, write_graph=True)

    adv_model_filename = r'./runs/{}/adv_{}.dat'.format(dataset, args.model)
    adv_model_path = os.path.split(adv_model_filename)[0]
    if not os.path.exists(adv_model_path):
        os.makedirs(adv_model_path)
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

    print('Adversarial Training...')
    print('batch_size: ', batch_size, "; epochs: ", epochs)
    model.fit(x_train_plus, y_train_plus,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tb_callback])
    scores = model.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    print('Saving model weights...')
    model.save_weights(adv_model_filename)


if __name__ == '__main__':
    args = parser.parse_args()
    adversarial_training()
