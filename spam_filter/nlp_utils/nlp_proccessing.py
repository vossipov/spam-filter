from collections import defaultdict
from typing import List

import numpy as np
from scipy.sparse import dok_matrix

from .tokenizer import Tokenizer


def tokenize_corpus(source: List[str], tokenizer: Tokenizer, **tokenizer_kwargs):
    return [tokenizer.tokenize(text, **tokenizer_kwargs) for text in source]


def build_vocabulary(texts: List[str], max_size=100000, max_doc_freq=0.8, min_count=5, pad_word=None):
    word_counts = defaultdict(int)
    doc_n = 0

    for text in texts:
        doc_n += 1
        unique_text_words = set(text)
        for word in unique_text_words:
            word_counts[word] += 1

    word_counts = {word: cnt for word, cnt in word_counts.items()
                   if cnt >= min_count and cnt / doc_n <= max_doc_freq}

    sorted_word_counts = sorted(word_counts.items(),
                                reverse=True,
                                key=lambda pair: pair[1])

    if pad_word is None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    if len(word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    words_id = {word: i for i, (word, _) in enumerate(sorted_word_counts)}
    words_freq = np.array([cnt / doc_n for _, cnt in sorted_word_counts], dtype='float32')

    return words_id, words_freq


def vectorize_word(texts: List[str], words_id, words_freq, mode='tfidf', scale=True):
    assert mode in {'tf', 'idf', 'tfidf', 'bin'}
    matrix = dok_matrix((len(texts), len(words_id)), dtype='float32')

    for i, text in enumerate(texts):
        for word in text:
            if word in words_id:
                matrix[i, words_id[word]] += 1

    if mode == 'tf':
        matrix = matrix.tocsr()
        matrix = matrix.multiply(1 / matrix.sum(1))
    elif mode == 'idf':
        matrix = (matrix > 0).astype('float32').multiply(1 / words_freq)

    elif mode == 'bin':
        matrix = (matrix > 0).astype('float32')

    elif mode == 'tfidf':
        matrix = matrix.tocsr()
        matrix = matrix.multiply(1 / matrix.sum(1))
        matrix = matrix.multiply(1 / words_freq)

    # MinMax standardization
    if scale:
        matrix = matrix.tocsc()
        matrix -= matrix.min()
        matrix /= (matrix.max() + 1e-6)

    return matrix.tocsr()
