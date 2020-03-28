from abc import ABC, abstractmethod
import re
from typing import List

TOKEN_RE = re.compile(r'[\w+]')


class Tokenizer(ABC):

    @abstractmethod
    def tokenize(self, text: str, min_token_size: int):
        """
            Method should extract words and lead to the desired form of words
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
