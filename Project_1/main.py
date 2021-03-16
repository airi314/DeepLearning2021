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

# Read and show data
train = read_data('data/data.three_gauss.train.10000.csv', classification=True)
x, y = split_data(train, 2)
test = read_data('data/data.three_gauss.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)

plot_2d_data(x, y, "Training data")


# Choose network type and architecture
network = MLP([5], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
              alpha=0.9, max_epochs=5, regression=False, random_state=1)


# Train the network
network.fit(x, y, plot_arch=True, plot_errors_arch=True,
            evaluation_dataset=[x_test, y_test])
print("Accuracy on training data: " + str(accuracy(network.predict(x), y)))


# Measure performance on test set
plot_2d_data(x_test, network.predict(x_test), "Prediction on test data")
plot_2d_error(x_test, y_test, network.predict(x_test))
print("Accuracy on test data: " + str(accuracy(network.predict(x_test), y_test)))


# Final architecture
plot_architecture(network.neurons, [l.W.T for l in network.layers])

