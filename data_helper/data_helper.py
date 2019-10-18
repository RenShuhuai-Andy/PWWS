from .read_files import split_imdb_files, split_yahoo_files, split_agnews_files
from .word_level_process import word_process
from .char_level_process import char_process
from sklearn.utils import shuffle


class DataHelper:
    def __init__(self, dataset, level):
        self.dataset = dataset
        self.level = level

        self.train_texts = None
        self.train_labels = None
        self.test_texts = None
        self.test_labels = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def load_data(self):
        if self.dataset == 'imdb':
            self.train_texts, self.train_labels, self.test_texts, self.test_labels = split_imdb_files()
        elif self.dataset == 'agnews':
            self.train_texts, self.train_labels, self.test_texts, self.test_labels = split_agnews_files()
        elif self.dataset == 'yahoo':
            self.train_texts, self.train_labels, self.test_texts, self.test_labels = split_yahoo_files()
        else:
            raise ValueError("Can not process this dataset.")  # TODO
        return self.train_texts, self.train_labels, self.test_texts, self.test_labels

    def processing(self, need_shuffle=True):
        if not all([self.train_texts, self.train_labels, self.test_texts, self.test_labels]):
            self.load_data()

        if self.level == 'word':
            self.x_train, self.y_train, self.x_test, self.y_test = word_process(self.train_texts, self.train_labels,
                                                                                self.test_texts, self.test_labels,
                                                                                self.dataset)
        elif self.level == 'char':
            self.x_train, self.y_train, self.x_test, self.y_test = char_process(self.train_texts, self.train_labels,
                                                                                self.test_texts, self.test_labels,
                                                                                self.dataset)
        else:
            raise ValueError("Processing level must be 'word' or 'char'.")

        if need_shuffle:
            self.x_train, self.y_train = shuffle(self.x_train, self.y_train, random_state=0)

        return self.x_train, self.y_train, self.x_test, self.y_test
