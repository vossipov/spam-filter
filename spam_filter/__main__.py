import numpy as np
import pandas as pd

from .nlp_utils.classifier import NaiveBayesClassifier
from .nlp_utils.tokenizer import NGramTokenizer

DATASET_PATH = 'spam_filter/data/spam.csv'


def preprocess_data():
    dataset = pd.read_csv(DATASET_PATH, encoding='latin-1')
    dataset.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
    dataset['label'] = dataset['labels'].map({'ham': 0, 'spam': 1})
    dataset.drop(['labels'], axis=1, inplace=True)

    train_indices, test_indices = [], []
    for i in range(dataset.shape[0]):
        if np.random.uniform(0, 1) < 0.75:
            train_indices += [i]
        else:
            test_indices += [i]

    train_dataset = dataset.loc[train_indices]
    test_dataset = dataset.loc[test_indices]

    train_dataset.reset_index(inplace=True)
    train_dataset.drop(['index'], axis=1, inplace=True)
    test_dataset.reset_index(inplace=True)
    test_dataset.drop(['index'], axis=1, inplace=True)
    return train_dataset, test_dataset


def metrics(labels, predictions):
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(labels)):
        true_pos += int(labels[i] == 1 and predictions[i] == 1)
        true_neg += int(labels[i] == 0 and predictions[i] == 0)
        false_pos += int(labels[i] == 0 and predictions[i] == 1)
        false_neg += int(labels[i] == 1 and predictions[i] == 0)

    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f_score = 2 * precision * recall / (precision + recall)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F-score: ", f_score)
    print("Accuracy: ", accuracy)


if __name__ == '__main__':
    train_dataset, test_dataset = preprocess_data()
    classifier = NaiveBayesClassifier()
    classifier.train(train_dataset)

    prediction_list = classifier.predict(test_dataset['message'])
    metrics(test_dataset['label'], prediction_list)

