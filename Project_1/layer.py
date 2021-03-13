import numpy as np
from activation import *


class Layer:
    def __init__(self, neurons_in, neurons_out, activ_function, random_state,
                 init="Xavier", bias_presence=True):

        self.neurons_in = neurons_in
        self.neurons_out = neurons_out
        self.af = activ_function
        self.bias_presence = bias_presence

        self.W = self.init_weights(neurons_in, neurons_out, random_state, init)
        self.mW = np.zeros([neurons_out, neurons_in])

        if self.bias_presence:
            self.bias = self.init_weights(1, neurons_out, random_state, init)
            self.m_bias = np.zeros([neurons_out, 1])

    def init_weights(self, neurons_in, neurons_out, random_state, init):
        r = np.random.RandomState(random_state)

        if init == 'zeros':
            return np.zeros([neurons_out, neurons_in])
        elif init == 'uniform':
            return r.uniform(0, 1, (neurons_out, neurons_in))
        elif init == 'Xavier':
            return r.randn(neurons_out, neurons_in)*np.sqrt(1/(neurons_in))
        elif init == "He":
            return r.randn(neurons_out, neurons_in)*np.sqrt(2/(neurons_in+neurons_out))

    def forward(self, x):
        self.forward_linear = self.W @ x
        if self.bias_presence:
            self.forward_linear += self.bias
        return self.af(self.forward_linear)

    def backpropagate_last_layer(self, error_weight):
        self.forward_gradient = self.af(self.forward_linear, derivative=True)
        if self.af != softmax:
            self.backward_error = error_weight * self.forward_gradient
        else:
            ex = np.exp(self.forward_linear)
            sum_ex = np.sum(ex, axis=0)
            gradient = sum_ex * ex / (sum_ex) ** 2
            self.backward_error = error_weight * gradient
            for i in range(self.forward_linear.shape[0]):
                self.backward_error[i, :] += np.sum(
                    error_weight * self.forward_gradient, axis=0) * ex[i, :]

    def backpropagate_layer(self, error_weight):
        self.forward_gradient = self.af(self.forward_linear, derivative=True)
        self.backward_error = error_weight.T * self.forward_gradient

    def error_weight(self):
        return self.backward_error.T @ self.W

    def update_weights(self, previous_predict, eta, alpha):
        dW = self.backward_error @ previous_predict.T
        self.mW = (1-alpha) * dW + alpha * self.mW
        self.W -= eta * self.mW

        if self.bias_presence:
            d_bias = np.sum(self.backward_error)
            self.m_bias = d_bias + alpha * self.m_bias
            self.bias -= eta * self.m_bias
