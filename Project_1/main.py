# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# %%
from activation import *
from evaluation import *
from utils import read_data, split_data, scale
from plots import *

# %%
from layer import Layer
from mlp import MLP

# %%
train = read_data('data/data.simple.train.10000.csv', classification=True)
x,y = split_data(train, 2)
test = read_data('data/data.simple.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)

# %%
plot_2d_data(x, y, "Training data")

# %%
network = MLP([20], sigmoid, init= 'Xavier', bias_presence = True, eta=0.01,
              alpha=0.9, max_epochs=100, regression=False)

# %%
network.fit(x,y, x_test, y_test)
plot_errors_vs_epochs(network.errors, network.errors_test, "Cross-entropy")

# %%
plot_2d_data(x_test, network.predict(x_test), "Prediction on test data")
print("Accuracy on test data: " + str(accuracy(network.predict(x_test, predict_proba=True), y_test)))

