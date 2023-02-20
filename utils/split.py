import numpy as np

def split_n_folds(n_folds: int, messages: list, labels:list, shuffled_idx: list) -> tuple:
    """Split the data into n_folds folds."""
    training_folds_el = []
    training_folds_labels = []
    offest = (len(shuffled_idx)//n_folds)
    for i in range(n_folds):
        training_folds_el.append([messages[j] for j in shuffled_idx[offest*i:offest*(i+1)]])
        training_folds_labels.append([labels[j] for j in shuffled_idx[offest*i:offest*(i+1)]])
        
    return training_folds_el , training_folds_labels

def shuffle_idx(len_labels: int) -> list:
    """Shuffle the index of the elements."""
    idx = np.linspace(0, len_labels, num=len_labels, endpoint=False, dtype='int')
    np.random.shuffle(idx)
    return idx