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
train = read_data('data/data.three_gauss.train.10000.csv', classification=True)
x, y = split_data(train, 2)
test = read_data('data/data.three_gauss.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)

# %%
# plot_2d_data(x, y, "Training data")

# %%
print('Default measure - cross entropy for classification with softmax function in last layer')
network = MLP([20], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
              alpha=0.9, max_epochs=100, regression=False, random_state=1)

# %%
# network.fit(x, y, plot_arch=True, plot_errors_arch=True,
#             evaluation_dataset=[x_test, y_test])
network.fit(x, y, evaluation_dataset=[x_test, y_test], calculate_accuracy=True)

#plot_errors_vs_epochs(network.errors, network.errors_test, "Cross-entropy")
#plot_2d_error(x_test, y_test, network.predict(x_test))

# print(network.errors)
# print(network.errors_test)


# %%
# plot_2d_data(x_test, network.predict(x_test), "Prediction on test data")
print("Accuracy on training data: " + str(accuracy(network.predict(x),y)))
print("Accuracy on test data: " + str(accuracy(network.predict(x_test),y_test)))

# %%
print('Measure - binary cross entropy with sigmoid function in last layer')
network = MLP([20], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
              alpha=0.9, max_epochs=100, regression=False, random_state=1, measure='binary_cross_entropy')
network.fit(x, y, evaluation_dataset=[x_test, y_test], calculate_accuracy=True)
print("Accuracy on training data: " + str(accuracy(network.predict(x),y)))
print("Accuracy on test data: " + str(accuracy(network.predict(x_test),y_test)))

# %%
#plot_architecture(network.neurons, [l.W.T for l in network.layers])



# architectures = [[i] for i in range(1, 11)]
# print(architectures)
# for arch in architectures:
#     print("Architecture: " + str(arch))
#     network = MLP(arch, sigmoid, init='Xavier', bias_presence=True, eta=0.01,
#                   alpha=0.9, max_epochs=100, regression=False, random_state=1)
#     network.fit(x, y, measure='cross_entropy', plot_errors=False, evaluation_dataset=[x_test, y_test])
#     print("Accuracy on test data: " + str(accuracy(network.predict(x_test, predict_proba=True), y_test)))