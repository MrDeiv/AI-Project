import numpy as np
from sklearn.feature_selection import mutual_info_classif

def join_in_set(messages: list) -> set:
    """Join the lists inside messages into a single set."""
    training_set = set()
    for el in messages:
        training_set.update(el)
    return training_set

def compute_feature_vector(training_set: dict, messages: list, matrix: np.ndarray) -> np.ndarray:
    m_copy = matrix.copy()
    for i,el in enumerate(messages):
        for word in el:
            m_copy[i, training_set[word]] += 1
    return m_copy

def get_n_words_feature(words: list, matrix: np.ndarray, messages: list, n_words: int) -> dict:
    """Get the n_words with the highest mutual information."""
    res = dict(zip(words, mutual_info_classif(matrix, messages, discrete_features=True)))
    res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True)[:n_words])
    return res