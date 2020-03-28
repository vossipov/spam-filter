import pandas as pd

DATASET_PATH = 'data/spam.csv'


def read_dataset(path: str = DATASET_PATH):
    raw_dataset = pd.read_csv(path, encoding='latin-1')
    dataset = raw_dataset.where((pd.notnull(raw_dataset)), '')

    dataset.loc[dataset["Category"]]
