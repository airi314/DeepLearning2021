import numpy as np
import pandas as pd


def scale(X):
    return (X-np.mean(X, 0))/np.std(X, 0)


def split_data(data, input_colums):
    x = data[:, 0:input_colums]  # inputs
    y = data[:, input_colums:(input_colums+1)]  # outputs
    return x, y


def read_data(file, classification=False):
    data = np.genfromtxt(file, delimiter=',')
    if not classification:
        data = data[1:, 1:]
    else:
        data = data[1:, :]
    return data


def one_hot_encode(y):
    y = pd.DataFrame(y)
    y = pd.get_dummies(y[0])
    return np.array(y)


def shuffle_data(x, y, random_state):
    r = np.random.RandomState(random_state)
    t = np.arange(x.shape[0])
    r.shuffle(t)
    return x[t], y[t]


def split_batches(x, y, batch_size):
    batch_nr = np.shape(x)[0]//batch_size
    batch_x = np.array_split(x, batch_nr)
    batch_y = np.array_split(y, batch_nr)
    return batch_x, batch_y
