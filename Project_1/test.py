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


## Testing activation functions

train = read_data('data/data.simple.train.10000.csv', classification=True)
x, y = split_data(train, 2)
test = read_data('data/data.simple.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)

# Accuracy throughout learning process for one random state
network = MLP([5], sigmoid, alpha=0.9, max_epochs=100,
              regression=False, random_state=1, epochs_no_change=100)
network.fit(x, y, evaluation_dataset=[
            x_test, y_test], plot_errors=False, calculate_accuracy=True)
acc_sigm = network.accuracy

network = MLP([5], relu, alpha=0.9, max_epochs=100,
              regression=False, random_state=1, epochs_no_change=100)
network.fit(x, y, evaluation_dataset=[
            x_test, y_test], plot_errors=False, calculate_accuracy=True)
acc_relu = network.accuracy

network = MLP([5], tanh, alpha=0.9, max_epochs=100,
              regression=False, random_state=1, epochs_no_change=100)
network.fit(x, y, evaluation_dataset=[
            x_test, y_test], plot_errors=False, calculate_accuracy=True)
acc_tanh = network.accuracy

plt.figure()
x = [x for x in range(100)]
plt.plot(x, acc_sigm, 'red')
plt.plot(x, acc_relu, 'blue')
plt.plot(x, acc_tanh, 'k')
plt.title("Accuracy during learning for different activation functions")
plt.legend(["Sigmoid", "ReLU", "Hyperbolic tangent"])
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig('act_fn_acc.png')

# Mean accuracy for different random states
n = 100
act_fn = [sigmoid, relu, tanh]
acc_tr = np.zeros((n, len(act_fn)))
acc_tst = np.zeros((n, len(act_fn)))
for i in range(n):
    print(i)
    for idx, fn in enumerate(act_fn):
        network = MLP([5], fn, alpha=0.9, max_epochs=100,
                      regression=False, random_state=i)
        network.fit(x, y, evaluation_dataset=[
                    x_test, y_test], plot_errors=False)
        acc_tr[i][idx] = accuracy(network.predict(x), y)
        acc_tst[i][idx] = accuracy(network.predict(x_test), y_test)
print('Train and test acc for sigmoid, relu and tanh')
print(np.mean(acc_tr, axis=0))
print(np.mean(acc_tst, axis=0))


## Comparing different networks architectures

colors = ['r', 'g', 'b', 'y', 'm', 'black']
train = read_data('data/data.three_gauss.train.10000.csv', classification=True)
x, y = split_data(train, 2)
test = read_data('data/data.three_gauss.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)

# One hidden layer, different number of neurons
accuracy_list = list()
architectures = [[5], [10], [25], [100], [200], [500]]
for arch in architectures:
    network = MLP(arch, sigmoid, init='Xavier', bias_presence=True, eta=0.01,
                  alpha=0.9, max_epochs=150, regression=False, random_state=1, epochs_no_change=10)
    network.fit(x, y, evaluation_dataset=[
                x_test, y_test], calculate_accuracy=True, plot_errors=False)
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
    network.fit(x, y, evaluation_dataset=[
                x_test, y_test], calculate_accuracy=True, plot_errors=False)
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


## Comparing different loss functions

train = read_data('data/data.simple.train.10000.csv', classification=True)
x, y = split_data(train, 2)
test = read_data('data/data.simple.test.10000.csv', classification=True)
x_test, y_test = split_data(test, 2)
y = y - [1]
y_test = y_test - [1]
x = scale(x)
x_test = scale(x_test)

# Default measure - cross entropy for classification with softmax function in last layer
network_1 = MLP([20], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
                alpha=0.9, max_epochs=300, epochs_no_change=10, regression=False, random_state=1)

network_1.fit(x, y, evaluation_dataset=[
              x_test, y_test], calculate_accuracy=True, plot_errors=False)

# Measure - binary cross entropy with sigmoid function in last layer
network_2 = MLP([20], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
                alpha=0.9, max_epochs=300, epochs_no_change=10, regression=False,
                random_state=1, measure='binary_cross_entropy')

network_2.fit(x, y, evaluation_dataset=[
              x_test, y_test], calculate_accuracy=True, plot_errors=False)

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


## Quick tests on regression data

train = read_data('data/steps-small-training.csv', classification=False)
train = scale(train)
x, y = split_data(train, 1)
test = read_data('data/steps-small-test.csv', classification=False)
test = scale(test)
x_test, y_test = split_data(test, 1)


network = MLP([20], sigmoid, init='Xavier', bias_presence=True, eta=0.01,
              alpha=0.9, max_epochs=300, epochs_no_change=20, regression=True, random_state=1)

network.fit(x, y, evaluation_dataset=[x_test, y_test], plot_errors=False)

plot_errors_vs_epochs(network.errors, network.errors_test, 'MSE')
