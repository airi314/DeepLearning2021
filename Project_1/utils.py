import numpy as np
import pandas as pd

def scale(X):
    return (X-np.mean(X, 0))/np.std(X, 0)

def one_hot_encode(y):
    y = pd.DataFrame(y)
    y = pd.get_dummies(y[0])
    return np.array(y)

def shuffle_data(x, y):
    t = np.arange(x.shape[0])
    np.random.shuffle(t)
    return x[t], y[t]

def split_batches(x, y, bs):
    batch_nr = np.shape(x)[0]//bs
    batch_x = np.array_split(x, batch_nr)
    batch_y = np.array_split(y, batch_nr)
    return batch_x, batch_y
