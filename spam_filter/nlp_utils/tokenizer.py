from abc import ABC, abstractmethod
import re
from typing import List
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

TOKEN_RE = re.compile(r'[\w+]')


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str, min_token_size: int):
        """
            Method should extract words and lead them to the desired form
        :param text: to be processed
        :param min_token_size:
        :return: tokenized text
        """
        pass


class SimpleTokenizer(Tokenizer):
    """
        Simple word tokenizer
        which extract continuous sequence of letters or numbers
    """

    def tokenize(self, text: str, min_token_size: int = 4):
        txt = text.lower()
        token_list: List[str] = TOKEN_RE.findall(txt)
        return [token for token in token_list if len(token) >= min_token_size]


class StemmerTokenizer(Tokenizer):
    """
        Tokenizer that use Porter stemmer in order to lead words to 'normal form'
        thereby reduce the dimension of dictionary
    """

    def tokenize(self, text: str, min_token_size: int):
        text = text.lower()
        words = word_tokenize(text)
        words = [w for w in words if len(w) > min_token_size]

        stop_words = stopwords.words('english')
        words = [w for w in words if words not in stop_words]

        stemmer = PorterStemmer()
        words = [stemmer.stem(w) for w in words]

        return words
