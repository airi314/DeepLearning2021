import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from activation import *
from evaluation import *
from utils import read_data, split_data, scale
from plots import *

from layer import Layer
from mlp import MLP

colors = ['r', 'g', 'b', 'y', 'm', 'black']
train = read_data('data/data.three_gauss.train.10000.csv', classification=True)
x, y = split_data(train, 2)
test = read_data('data/data.three_gauss.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)

# Comparing different networks architectures

# One hidden layer, different number of neurons
accuracy_list = list()
architectures = [[5], [10], [25], [100], [200], [500]] 
for arch in architectures:
    network = MLP(arch, sigmoid, init='Xavier', bias_presence=True, eta=0.01,
                  alpha=0.9, max_epochs=150, regression=False, random_state=1, epochs_no_change=10)
    network.fit(x, y, evaluation_dataset=[x_test, y_test], calculate_accuracy=True, plot_errors=False)
    accuracy_list.append(network.accuracy)

plt.figure()
for i, acc in enumerate(accuracy_list):
    epochs = [x for x in range(len(acc))]
    plt.plot(epochs, acc, colors[i])

plt.legend(architectures)
plt.title("Accuracy depending on network architecture")
plt.ylim([0.93, 0.94])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig('arch_one_layer.png')


# Different number of hidden layers
accuracy_list = list()
architectures = [[10], [10, 10], [10, 10, 10], [10, 10, 10, 10]] 
for arch in architectures:
    network = MLP(arch, sigmoid, init='Xavier', bias_presence=True, eta=0.01,
                  alpha=0.9, max_epochs=150, regression=False, random_state=1, epochs_no_change=10)
    network.fit(x, y, evaluation_dataset=[x_test, y_test], calculate_accuracy=True, plot_errors=False)
    accuracy_list.append(network.accuracy)

plt.figure()
colors = ['r', 'g', 'b', 'y', 'm', 'black']
for i, acc in enumerate(accuracy_list):
    epochs = [x for x in range(len(acc))]
    plt.plot(epochs, acc, colors[i])

plt.legend(architectures)
plt.title("Accuracy depending on network architecture")
plt.ylim([0.93, 0.94])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig('arch_many_layers.png')


train = read_data('data/data.simple.train.10000.csv', classification=True)
x, y = split_data(train, 2)
test = read_data('data/data.simple.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)


# Comparing different loss functions
# Default measure - cross entropy for classification with softmax function in last layer
network_1 = MLP([20], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
              alpha=0.9, max_epochs=300, epochs_no_change=10, regression=False, random_state=1)

network_1.fit(x, y, evaluation_dataset=[x_test, y_test], calculate_accuracy=True, plot_errors=False)

# Measure - binary cross entropy with sigmoid function in last layer
network_2 = MLP([20], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
                alpha=0.9, max_epochs=300, epochs_no_change=10, regression=False, 
                random_state=1, measure='binary_cross_entropy')

network_2.fit(x, y, evaluation_dataset=[x_test, y_test], calculate_accuracy=True, plot_errors=False)

plt.figure()
labels = ['cross entropy', 'binary cross entropy']
for i, acc in enumerate([network_1.accuracy, network_2.accuracy]):
    epochs = [x for x in range(len(acc))]
    plt.plot(epochs, acc, colors[i])
plt.legend(labels)
plt.title("Accuracy depending on loss function")
plt.ylim([0.99, 0.997])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig('cross_entropy.png')