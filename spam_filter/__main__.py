import pandas as pd

DATASET_PATH = './data/spam.csv'


def preprocess_data():
    dataset = pd.read_csv(DATASET_PATH, encoding='latin-1')
    dataset.rename(columns={'v1': 'labels', 'v2': 'message'}, inplace=True)
    dataset['label'] = dataset['labels'].map({'ham': 0, 'spam': 1})
    dataset.drop(['labels'], axis=1, inplace=True)
    return dataset
