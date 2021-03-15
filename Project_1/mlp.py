from activation import *
from utils import one_hot_encode, shuffle_data, split_batches
from layer import Layer
from evaluation import measure_performance
from plots import plot_architecture, plot_weight_updates, plot_errors_vs_epochs
from copy import deepcopy


class MLP:
    def __init__(self, hidden_layers, activ_function=sigmoid, batch_size=32,
                 init="Xavier", bias_presence=True, regression=True,
                 eta=0.01, alpha=0, max_epochs=100, epochs_no_change=3,
                 min_improvement=0.0001, random_state=123, measure=None):
        """
        Multi-layer Perceptron classifier.

        Parameters
        ----------
        hidden_layers : list of intl; length = n_layers - 2
            i-th place corresponds to i-th hidden layer size

        activ_function : {linear, sigmoid, tanh, relu}, default=sigmoid
            Ativation function for hidden layers.

        batch_size : int
            Size of the batch during the training

        init : {'uniform', 'linear', 'Xavier' 'He'}, default='Xavier'
            Method of weights initialization

        bias_presence : boolean; default: True
            If bias is present in the training process

        regression : boolean; default: True
            If problem is regression problem or not

        eta : float, default: 0.01
            Learning rate

        alpha : float, default: 0
            Momentum for gradient descent update; should be between 0 and 1; if 0 then inactivate

        max_epochs : int; default: 500
            Maximum number of epochs

        epochs_no_change : int; default: 20
            Maximum number of epochs with no improvement

        random_state : int; default: 123
            Random state using during training

        """

        self.random_state = random_state
        self.hidden_layers = hidden_layers
        self.activ_function = activ_function
        self.init = init
        self.bias_presence = bias_presence
        self.initialized = False

        self.batch_size = batch_size
        self.regr = regression


        self.measure = measure
        if self.measure is None and self.regr:
            self.measure = "MSE"
        elif self.measure is None:
            self.measure = "cross_entropy"
        self.last_activ = linear if self.regr else softmax if self.measure=='cross_entropy' else sigmoid

        self.eta = eta
        self.alpha = alpha

        self.max_epochs = max_epochs
        self.epochs_no_change = epochs_no_change
        self.min_improvement = min_improvement

    def __create_layers(self, neurons_x, neurons_y):

        self.layers = []
        self.neurons = [neurons_x] + self.hidden_layers + [neurons_y]

        for i in range(len(self.neurons)-2):
            self.layers.append(Layer(self.neurons[i], self.neurons[i+1],
                                     self.activ_function, self.random_state, self.init, self.bias_presence))
        self.layers.append(Layer(self.neurons[i+1], self.neurons[i+2],
                                 self.last_activ, self.random_state, self.init, self.bias_presence))
        self.initialized = True

    def __forward(self, x):
        self.z = [0]*len(self.layers)
        x = x.T
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            self.z[i] = x
        return x.T

    def __backpropagate(self, x, y):

        x, y = shuffle_data(x, y, self.random_state)
        batch_x, batch_y = split_batches(x, y, self.batch_size)

        self.__save_weights()
        for b_x, b_y in zip(batch_x, batch_y):
            self.__backpropagate_batch(b_x, b_y)

    def __save_weights(self):
        for i in range(len(self.layers)-1, -1, -1):
            self.layers[i].prev_W = deepcopy(self.layers[i].W)

    def __backpropagate_batch(self, x, y):
        z = self.__forward(x)

        error_weight = z.T - y.T

        self.layers[-1].backpropagate_last_layer(error_weight)
        for i in range(len(self.layers)-2, -1, -1):
            error_weight = self.layers[i+1].error_weight()
            self.layers[i].backpropagate_layer(error_weight)
        self.__update_weights(x)

    def __update_weights(self, x):
        self.layers[0].update_weights(x.T, self.eta, self.alpha)
        for i, layer in enumerate(self.layers[1:]):
            layer.update_weights(self.z[i], self.eta, self.alpha)

    def __compute_errors(self, x, y, evaluation_dataset, calculate_accuracy):

        if self.regr:
            self.measure = "MSE" if self.measure is None else self.measure
        else:
            self.measure = "accuracy" if self.measure is None else self.measure

        self.best_error = np.inf if self.best_error is None else self.best_error

        y_pred = self.__forward(x)
        train_error = measure_performance(y_pred, y, self.regr, self.measure)
        self.errors.append(train_error)

        if calculate_accuracy:
            self.accuracy.append(measure_performance(np.argmax(y_pred, axis=1), np.argmax(y, axis=1), self.regr, "accuracy"))

        if evaluation_dataset:
            x_test, y_test = evaluation_dataset
            y_test_pred = self.__forward(x_test)
            self.errors_test.append(measure_performance(
                y_test_pred, y_test, self.regr, self.measure))

        if train_error < self.best_error - self.min_improvement:
            self.best_error = train_error
            self.count_no_change = 0
        else:
            self.count_no_change += 1

    def fit(self, x, y, evaluation_dataset=None, plot_errors=True, plot_arch=False, plot_errors_arch=False, calculate_accuracy=False):

        self.errors = []
        self.errors_test = []
        self.best_error = None
        self.count_no_change = 0

        if not self.regr:
            y = one_hot_encode(y)

        if not self.regr and evaluation_dataset:
            evaluation_dataset[1] = one_hot_encode(evaluation_dataset[1])

        if calculate_accuracy:
            self.accuracy = []

        self.__create_layers(x.shape[1], y.shape[1])
        self.__compute_errors(x, y, evaluation_dataset, calculate_accuracy)

        if plot_arch:
            plot_architecture(self.neurons, [l.W.T for l in self.layers])

        for i in range(self.max_epochs):
            self.__backpropagate(x, y)
            self.__compute_errors(x, y, evaluation_dataset, calculate_accuracy)

            if plot_errors_arch:
                plot_weight_updates(
                    self.neurons, [l.W.T - l.prev_W.T for l in self.layers])

            if plot_arch:
                plot_architecture(self.neurons, [l.W.T for l in self.layers])

            if self.count_no_change > self.epochs_no_change:
                print('Stop training process after ' + str(i) + " epochs. There were " +
                      str(self.epochs_no_change) + ' epochs with no improvement.')
                break

        if plot_errors:
            plot_errors_vs_epochs(self.errors, self.errors_test, self.measure)

    def predict(self, x, predict_proba=False):
        y_pred = self.__forward(x)
        if not self.regr and not predict_proba:
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = y_pred.reshape([-1, 1])
        return y_pred

