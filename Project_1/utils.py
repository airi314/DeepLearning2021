import numpy as np
import pandas as pd
import os
import torch


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

def load_mnist(dir_path):
    [x, y] = torch.load(os.path.join(dir_path, 'processed/training.pt'))
    [x_test, y_test] = torch.load(os.path.join(dir_path, 'processed/test.pt'))
    
    x = np.array(x).reshape(60000, -1)/255
    y = np.array(y).reshape(-1,1)
    x_test = np.array(x_test).reshape(10000, -1)/255
    y_test = np.array(y_test).reshape(-1,1)
    
    return x, y, x_test, y_test