# main file
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import string
import time

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer


def remove_punctuation(message: str) -> str:
    """Remove punctuation from a message."""
    return "".join([c for c in message if c not in string.punctuation])


def remove_stopwords(message: str, stop_word: set) -> list:
    """Tokenize and remove stopwords from a message."""
    return [word for word in word_tokenize(message.lower()) if word not in stop_word]


def stem_messages(message: list, stemmer: PorterStemmer) -> list:
    """Stem a list of messages."""
    return [stemmer.stem(el) for el in message]


def sanitize_messages(messages: list) -> list:
    """Remove punctuation, tokenize and remove stopwords from a list of messages."""
    stemmer = PorterStemmer()
    stop_word = set(stopwords.words('english'))
    return [stem_messages(remove_stopwords(remove_punctuation(message), stop_word), stemmer) for message in messages]


if __name__ == '__main__':

    #TODO: check if nltk packages are installed
    nltk.download('punkt')
    nltk.download('stopwords')

    # import the complete dataset
    newsgroup_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    # get label of each message
    labels = newsgroup_data.target

    # get messages
    start_t = time.time()

    # ITEMS 1 AND 2
    ready_messages = sanitize_messages(newsgroup_data.data)

    end_t = time.time()
    print(f"Time to sanitize messages: {format(end_t - start_t, '.2f')} seconds")

    # ITEMS 3
    # Split the data into 5 fold
    idx = np.linspace(0, len(labels), num=len(labels), endpoint=False, dtype='int')
    np.random.shuffle(idx)  # shuffling the elements of idx (in-place)

    training_folds_el = []
    training_folds_labels = []
    offest = (len(idx)//5)
    for i in range(3):
        training_folds_el.append([ready_messages[j] for j in idx[offest*i:offest*(i+1)]])
        training_folds_labels.append([labels[j] for j in idx[offest*i:offest*(i+1)]])

    test_fold_el = [ready_messages[j] for j in idx[offest*4:]]
    test_fold_labels = [labels[j] for j in idx[offest*4:]]

    # ITEM 3.1
    # join the lists inside training_folds_el[0] into a single set
    training_set = set()
    for el in training_folds_el[0]:
        training_set.update(el)
    
    # transform training_set into a dictionary with value corresponding to the index of the word
    training_set = dict(zip(training_set, range(len(training_set))))

    # create a matrix:
    # - rows: number of elements in training_folds_el[0]
    # - cols: number of words in training_set
    training_matrix = np.zeros((len(training_folds_el[0]), len(training_set)))

    # count the number of occurences of each word in each element of training_folds_el[0]
    for i, el in enumerate(training_folds_el[0]):
        for word in el:
            training_matrix[i, training_set[word]] += 1
    np.set_printoptions(threshold=1000)
    print(training_matrix[0])
        






