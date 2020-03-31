from collections import defaultdict
from typing import List


def tokenize_corpus(source: List[str], tokenizer, min_token_size):
    return [tokenizer.tokenize(text, min_token_size) for text in source]


def build_vocabulary(texts, max_doc_freq=0.8, min_count=3):
    word_counts = defaultdict(int)

    for text in texts:
        unique_text_words = set(text)
        for word in unique_text_words:
            word_counts[word] += 1

    doc_n = len(word_counts)

    word_counts = {word: cnt for word, cnt in word_counts.items()
                   if cnt >= min_count and cnt / doc_n <= max_doc_freq}

    return word_counts
