import numpy as np
from utils import one_hot_encode

def MSE(y_pred, y):
    return np.mean((y_pred - y) ** 2)
    
def MAE(y_pred, y):
    return np.mean(np.absolute(y_pred-y))
    
def cat_cross_entropy(y_pred, y):
    return -(np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred)))/y.shape[0]

def hinge_loss(y_pred, y):
    pass

def accuracy(y_pred, y):
    y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred.reshape([-1,1])
    return np.sum(y==y_pred)/y.shape[0]