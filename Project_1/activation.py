import numpy as np

def sigmoid(x, derivative=False):
    if derivative:
        return np.exp(x) / ((1 + np.exp(x)) ** 2)
    return 1/(1+np.exp(-x))

def linear(x, derivative=False):
    if derivative:
        return np.ones(x.shape)
    return x

def softmax(x, derivative=False):
    if derivative:
        return -np.e**x / (np.sum(np.e**x, axis=0) ** 2)
    return np.e**x / np.sum(np.e**x, axis=0)

def tanh(x, derivative=False):
    e2x = (np.exp(2*x))
    if derivative:
        return (e2x * 4) / ((e2x + 1) ** 2)
    return (e2x - 1) / (e2x + 1)

def relu(x, derivative=False):
    if derivative:
        return np.maximum(np.sign(x) * x / x, 0)
    return np.maximum(x, 0)
