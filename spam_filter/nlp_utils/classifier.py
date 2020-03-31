from abc import ABC, abstractmethod

from .nlp_proccessing import tokenize_corpus, build_vocabulary
from .tokenizer import StemmerTokenizer

MIN_TOKEN_SIZE = 2
SPAM_LABEL = 1
HAM_LABEL = 0


class Classifier(ABC):
    """
        Abstract class for all classifiers.

        There are two main methods that should be implemented,
        also you can add methods for computing some metrics.
    """

    @abstractmethod
    def train(self, training_dataset):
        """
            Method that train model.
        :param training_dataset:
        :return:
        """
        pass

    @abstractmethod
    def classify(self, message):
        """
            Method that will be using to classify given message
            according to training model
        :param message: to be classified
        :return:
        """
        pass


class NaiveBayesClassifier(Classifier):
    """
        Naive Bayes Classifier implementation
    """

    def __init__(self):
        self.spam_prob = 0  # P(spam)
        self.ham_prob = 0  # P(ham)
        self.words_prob_in_spam = None
        self.words_prob_in_ham = None
        self.words_prob_in_dataset = None
        self.spam_vocab = None
        self.ham_vocab = None
        self.number_of_words_in_dataset = 0
        self.number_of_words_in_spam = 0
        self.number_of_words_in_ham = 0

        self.tokenizer = StemmerTokenizer()
        self.alpha = 1

    def train(self, training_set):
        spam_mails = list(training_set[training_set['label'] == SPAM_LABEL]['message'])
        ham_mails = list(training_set[training_set['label'] == HAM_LABEL]['message'])

        tokenized_spam_corpus = tokenize_corpus(spam_mails, tokenizer=self.tokenizer, min_token_size=MIN_TOKEN_SIZE)
        tokenized_ham_corpus = tokenize_corpus(ham_mails, tokenizer=self.tokenizer, min_token_size=MIN_TOKEN_SIZE)

        self.number_of_words_in_spam = sum([len(txt) for txt in tokenized_spam_corpus])
        self.number_of_words_in_ham = sum([len(txt) for txt in tokenized_ham_corpus])
        self.number_of_words_in_dataset = self.number_of_words_in_ham + self.number_of_words_in_spam

        self.spam_vocab = build_vocabulary(tokenized_spam_corpus)
        self.ham_vocab = build_vocabulary(tokenized_ham_corpus)

        self.spam_prob = len(spam_mails) / float(len(training_set['message']))
        self.ham_prob = len(ham_mails) / float(len(training_set['message']))

    def classify(self, message):
        tokenized_message = self.tokenizer.tokenize(message, min_token_size=MIN_TOKEN_SIZE)

        words_ham_prob = self._compute_prob(tokenized_message, False)
        words_spam_prob = self._compute_prob(tokenized_message, True)

        spam_prob = words_spam_prob * self.spam_prob
        ham_prob = words_ham_prob * self.ham_prob

        return spam_prob > ham_prob

    def predict(self, test_dataset):

        result = dict()
        for (i, message) in enumerate(test_dataset):
            result[i] = int(self.classify(message))
        return result

    def _word_prob(self, word, is_spam):
        if is_spam:
            return (self.spam_vocab.get(word, 0) + self.alpha) / (
                    self.number_of_words_in_spam + self.alpha * self.number_of_words_in_dataset)
        return (self.ham_vocab.get(word, 0) + self.alpha) / (
                self.number_of_words_in_ham + self.alpha * self.number_of_words_in_dataset)

    def _compute_prob(self, message, is_spam):
        result = 1.0
        for word in message:
            result *= self._word_prob(word, is_spam)
        return result
