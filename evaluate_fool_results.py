from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
import argparse
from data_helper.data_helper import DataHelper
from data_helper.word_level_process import get_tokenizer, text_to_vector_for_all
from data_helper.char_level_process import doc_process_for_all, get_embedding_dict
from neural_networks import word_cnn, char_cnn, bd_lstm, lstm
import spacy
import tensorflow as tf
from keras import backend as K

nlp = spacy.load('en_core_web_sm')

# # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(
    description='Evaluate fool accuracy for a text classifier.')
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
                    default='imdb')
parser.add_argument('-l', '--level',
                    help='The level of process dataset',
                    choices=['word', 'char'],
                    default='word')


def read_adversarial_file(adversarial_text_path):
    adversarial_text = list(open(adversarial_text_path, "r", encoding='utf-8').readlines())
    # remove sub_rate and NE_rate at the end of the text
    adversarial_text = [re.sub(' sub_rate.*', '', s) for s in adversarial_text]
    return adversarial_text


def get_mean_sub_rate(adversarial_text_path):
    adversarial_text = list(open(adversarial_text_path, "r", encoding='utf-8').readlines())
    all_sub_rate = []
    sub_rate_list = []
    for index, text in enumerate(adversarial_text):
        sub_rate = re.findall('\d+.\d+(?=; NE_rate)', text)
        if len(sub_rate) != 0:
            sub_rate = sub_rate[0]
            all_sub_rate.append(float(sub_rate))
            sub_rate_list.append((index, float(sub_rate)))
    mean_sub_rate = sum(all_sub_rate) / len(all_sub_rate)
    sub_rate_list.sort(key=lambda t: t[1], reverse=True)
    return mean_sub_rate


def get_mean_NE_rate(adversarial_text_path):
    adversarial_text = list(open(adversarial_text_path, "r", encoding='utf-8').readlines())
    all_NE_rate = []
    NE_rate_list = []
    for index, text in enumerate(adversarial_text):
        words = text.split(' ')
        NE_rate = float(words[-1].replace('\n', ''))
        all_NE_rate.append(NE_rate)
        NE_rate_list.append((index, NE_rate))
    mean_NE_rate = sum(all_NE_rate) / len(all_NE_rate)
    NE_rate_list.sort(key=lambda t: t[1], reverse=True)
    return mean_NE_rate


def process_adversarial_data(adv_text_filename, level, dataset, tokenizer):
    adv_text = read_adversarial_file(adv_text_filename)
    if level == 'word':
        return text_to_vector_for_all(adv_text, tokenizer, dataset)
    elif level == 'char':
        return doc_process_for_all(adv_text, get_embedding_dict(), dataset)
    else:
        raise ValueError("Processing level must be 'word' or 'char'.")


if __name__ == '__main__':
    args = parser.parse_args()
    clean_samples_cap = args.clean_samples_cap  # 1000

    # get tokenizer
    dataset = args.dataset
    tokenizer = get_tokenizer(dataset)

    # Load and process data set
    dataset = args.dataset
    data_helper = DataHelper(dataset, args.level)
    x_train = y_train = x_test = y_test = data_helper.processing()

    # Select the model and load the trained weights
    model = None
    if args.model == "word_cnn":
        model = word_cnn(dataset)
    elif args.model == "word_bdlstm":
        model = bd_lstm(dataset)
    elif args.model == "char_cnn":
        model = char_cnn(dataset)
    elif args.model == "word_lstm":
        model = lstm(dataset)
    model_filename = r'./runs/{}/{}.dat'.format(dataset, args.model)
    model.load_weights(model_filename)
    print('model path:', model_filename)

    # evaluate classification accuracy of model on clean samples
    scores_origin = model.evaluate(x_test[:clean_samples_cap], y_test[:clean_samples_cap])
    print('clean samples origin test_loss: %f, accuracy: %f' % (scores_origin[0], scores_origin[1]))
    all_scores_origin = model.evaluate(x_test, y_test)
    print('all origin test_loss: %f, accuracy: %f' % (all_scores_origin[0], all_scores_origin[1]))

    # Load and process adv data
    adv_text_filename = r'./fool_result/{}/{}/adv_{}.txt'.format(dataset, args.model, str(clean_samples_cap))
    print('adversarial file:', adv_text_filename)
    x_adv = process_adversarial_data(adv_text_filename, args.level, dataset, tokenizer)
    # evaluate classification accuracy of model on adversarial examples
    score_adv = model.evaluate(x_adv[:clean_samples_cap], y_test[:clean_samples_cap])
    print('adv test_loss: %f, accuracy: %f' % (score_adv[0], score_adv[1]))

    mean_sub_rate = get_mean_sub_rate(adv_text_filename)
    print('mean substitution rate:', mean_sub_rate)
    mean_NE_rate = get_mean_NE_rate(adv_text_filename)
    print('mean NE rate:', mean_NE_rate)
