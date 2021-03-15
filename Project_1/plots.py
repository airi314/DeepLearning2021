import numpy as np
import matplotlib.pyplot as plt
import VisualizeNN as VisNN


def plot_2d_data(x, y, plot_title="Figure 1"):
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
    plt.title(plot_title)
    plt.show()


def plot_errors_vs_epochs(errors, errors_test, measure_name):
    plt.figure()
    x = [x for x in range(len(errors))]
    plt.plot(x, errors, 'red')

    if errors_test:
        plt.plot(x, errors_test, 'blue')
        plt.title(measure_name +
                  " change on training and test sets during learning")
        plt.legend(["training set", "test set"])
    else:
        plt.title(measure_name + " change on training set during learning")

    plt.xlabel("Epochs")
    plt.ylabel(measure_name)
    plt.show()


def plot_2d_error(x, y, pred):
    plt.figure()
    y_new = np.array([0 if elem_y==elem_pred else 1 for elem_y, elem_pred in zip(y, pred)])
    uniques = np.unique(y_new)
    colors = ['g', 'r']
    for idx, elem in enumerate(uniques):
        ax = x[:, 0][y_new == elem]
        ay = x[:, 1][y_new == elem]
        plt.plot(ax, ay, colors[idx] + 'o')
    plt.legend(['Classified correctly', 'Classified incorrectly'])
    plt.title("Misclassified observations")
    plt.show()


def plot_architecture(neurons, weights):
    architect = VisNN.DrawNN(neurons, weights)
    architect.draw()


def plot_weight_updates(neurons, weights):
    architect = VisNN.DrawNN(neurons, weights, errors=True)
    architect.draw()
