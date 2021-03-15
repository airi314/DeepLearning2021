import numpy as np
from utils import one_hot_encode


def measure_performance(y_pred, y, regression=True, measure='accuracy'):

    measures_dict = {
        True:   {
            # 'MAE': MAE,
            'MSE': MSE
        },
        False:  {
            'accuracy': accuracy,
            'cross_entropy': cross_entropy,
            'binary_cross_entropy': bin_cross_entropy
        }
    }

    measure_function = measures_dict[regression][measure]
    error = measure_function(y_pred, y)

    return error


def MSE(y_pred, y):
    return np.mean((y_pred - y) ** 2)


# def MAE(y_pred, y):
#     return np.mean(np.absolute(y_pred-y))


def bin_cross_entropy(y_pred, y):
    return -(np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred)))/y.shape[0]


def cross_entropy(y_pred, y):
    return - np.sum(y * np.log(y_pred))/y.shape[0]


def accuracy(y_pred, y):
    return np.sum(y == y_pred)/y.shape[0]
