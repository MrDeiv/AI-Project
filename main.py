# main file
import numpy as np
from sklearn.datasets import fetch_20newsgroups
import time
import nltk

from utils.sanitize_messages import sanitize_messages
from utils.split import split_n_folds, shuffle_idx
from utils.features import join_in_set, compute_feature_vector, get_n_words_feature

if __name__ == '__main__':

    # download nltk data
    if not nltk.download('punkt'):
        nltk.download('punkt')
    
    if not nltk.download('stopwords'):
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
    idx = shuffle_idx(len(labels))
    training_folds_el, training_folds_labels = split_n_folds(5, ready_messages, labels, idx)

    result = []
    for tr_fold, tr_labels in zip(training_folds_el, training_folds_labels):
        # ITEMS 3.1, 3.2, 3.3
        # join the lists inside tr_fold into a single set
        training_set = join_in_set(tr_fold)
        
        # transform training_set into a dictionary with value corresponding to the index of the word
        training_set = dict(zip(training_set, range(len(training_set))))

        # create a matrix:
        # - rows: number of elements in tr_fold
        # - cols: number of words in training_set
        training_matrix = np.zeros((len(tr_fold), len(training_set)), dtype=int)

        # count the number of occurences of each word in each element of tr_fold
        training_matrix = compute_feature_vector(training_set, tr_fold, training_matrix)
        k = 100 #TODO: change this value for different k
        res = get_n_words_feature(training_set.keys(), training_matrix, tr_labels, k)
        result.append(res)

    print(result)





