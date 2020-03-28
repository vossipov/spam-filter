from abc import ABC, abstractmethod


class Classifier(ABC):
    """
        Abstract class for all classifiers
        There are two main methods that should be implemented
        also you can add methods for computing some metrics
    """

    @abstractmethod
    def train(self, training_dataset):
        """
            Method that train model
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

    def train(self, training_set):
        pass

    def classify(self, message):
        pass
