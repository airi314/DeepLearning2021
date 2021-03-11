import numpy as np
import matplotlib.pyplot as plt
import VisualizeNN as VisNN

def plot_2d_data(x, y, plot_title = "Figure 1"):
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

def plot_errors(errors, labels=[]):
    print(len(errors))
    colors = ['black', 'red', 'green', 'blue', 'yellow']
    plt.figure()
    for i, exp in enumerate(errors):
        plt.plot(exp, colors[i])
    legend = labels
    plt.legend(legend)
    plt.show()
    
def plot_errors_vs_epochs(errors, errors_test, measure_name):
    plt.figure()
    x = [x for x in range(len(errors))]
    plt.plot(x, errors, 'red')
    plt.plot(x, errors_test, 'blue')
    plt.title(measure_name + " change on training and test sets during learning")
    plt.legend(["training set", "test set"])
    plt.xlabel("Epochs")
    plt.ylabel(measure_name)
    plt.show()

def plot_2d_error(x, y, pred):
    plt.figure()
    y_new = np.abs(y-pred)
    uniques = np.unique(y_new[:, 0])
    colors = ['g', 'r']
    for i in range(len(uniques)):
        ax = x[:, 0][y_new[:, 0] == uniques[i]]
        ay = x[:, 1][y_new[:, 0] == uniques[i]]
        plt.plot(ax, ay, colors[i] + 'o')
    plt.legend(['Classified correctly', 'Classified incorrectly'])
    plt.title("Misclassified observations")
    plt.show()

def plot_architecture(neurons, weights):
    architect = VisNN.DrawNN(neurons, weights)
    architect.draw()
