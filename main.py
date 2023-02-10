# main file
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import string
import time

from sklearn.model_selection import StratifiedKFold, KFold


def remove_punctuation(message: str) -> str:
    """Remove punctuation from a message."""
    return "".join([c for c in message if c not in string.punctuation])


def remove_stopwords(message: str, stop_word: set) -> list:
    """Tokenize and remove stopwords from a message."""
    return [word for word in word_tokenize(message.lower()) if word not in stop_word]


def stem_messages(message: list, stemmer: PorterStemmer) -> list:
    """Stem a list of messages."""
    return list(set([stemmer.stem(el) for el in message]))


def sanitize_messages(messages: list) -> list:
    """Remove punctuation, tokenize and remove stopwords from a list of messages."""
    stemmer = PorterStemmer()
    stop_word = set(stopwords.words('english'))
    return [stem_messages(remove_stopwords(remove_punctuation(message), stop_word), stemmer) for message in messages]


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')

    # import the complete dataset
    newsgroup_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # get label of each message
    labels = newsgroup_data.target

    # get messages
    start_t = time.time()

    ready_messages = sanitize_messages(newsgroup_data.data)

    end_t = time.time()
    print(f"Time to sanitize messages: {format(end_t - start_t, '.2f')} seconds")

    # Split the data into 5 fold
    idx = np.linspace(0, len(labels), num=len(labels), endpoint=False, dtype='int')
    np.random.shuffle(idx)  # shuffling the elements of idx (in-place)

    print(len(labels), len(idx), len(ready_messages))
