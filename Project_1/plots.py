import numpy as np
import matplotlib.pyplot as plt

def plot_2d_data(x, y):
    plt.figure()
    uniques = np.unique(y[:, 0])
    colors = ['r', 'g', 'b']
    legend = []
    for i in range(len(uniques)):
        ax = x[:, 0][y[:, 0] == uniques[i]]
        ay = x[:, 1][y[:, 0] == uniques[i]]
        plt.plot(ax, ay, colors[i] + 'o')
        legend.append('class ' + str(int(uniques[i])))
    plt.legend(legend)
    plt.show()

def plot_errors(errors, labels=[]):
    colors = ['black', 'red', 'green', 'blue', 'yellow']
    plt.figure()
    for i, exp in enumerate(errors):
        plt.plot(exp, colors[i])
    legend = labels
    plt.legend(legend)
    plt.show()