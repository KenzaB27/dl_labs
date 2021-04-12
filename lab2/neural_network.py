import numpy as np
from typing import List
from enum import Enum
from gd_params import GDParams
import tqdm
import matplotlib.pyplot as plt


class ActivationFunction(Enum):
    RELU = 1
    SOFTMAX = 2


class Layer():
    def __init__(self, n_input, n_output, activation_function: ActivationFunction):
        self.W = np.random.normal(
            0, 1/np.sqrt(n_input), (n_output, n_input))
        self.b = np.zeros((n_output, 1))
        self.activation_function = activation_function
        self.last_input = None

    def softmax(self, x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def relu(self, x):
        return np.maximum(0.0, x)

    def activation(self, output):
        if self.activation_function == ActivationFunction.SOFTMAX:
            return self.softmax(output)
        elif self.activation_function == ActivationFunction.RELU:
            return self.relu(output)

    def evaluate_layer(self, input):
        self.last_input = input.copy()
        return self.activation(self.W @ input + self.b)

    def update_params(self, grad_W, grad_b, eta):
        self.W -= eta * grad_W
        self.b -= eta * grad_b


class NeuralNetwork():
    """Class representing Feed-Forward Neural Network suitable for classification tasks. """

    def __init__(self, layers: List[Layer], gd_params: GDParams):
        """ Class constructor

        Args:
            layers (list): list of layers
            gd_params (GDParams): parameters of gradient descent
        """
        self.layers = layers
        self.gd_params = gd_params
        self.eta = gd_params.eta
        self.n_batch = gd_params.n_batch
        self.lam = gd_params.lam
        self.n_epochs = gd_params.n_epochs

    def update_metrics(self, X, Y, X_val, Y_val, P, metrics):
        loss, cost = self.compute_cost(X, Y, P)
        v_loss, v_cost = self.compute_cost(X_val, Y_val)
        acc = self.compute_accuracy(X, Y)
        v_acc = self.compute_accuracy(X_val, Y_val)

        metrics['loss'].append(loss)
        metrics['val_loss'].append(v_loss)
        metrics['cost'].append(cost)
        metrics['val_cost'].append(v_cost)
        metrics['accuracy'].append(acc)
        metrics['val_accuracy'].append(v_acc)
        return metrics


    def compute_cost(self, X, Y, P=None):
        if P is None:
            P = self.evaluate_classifier(X)
        N = X.shape[1]
        loss = - np.sum(Y*np.log(P)) / N
        cost = loss + self.lam * sum([np.sum(layer.W**2) for layer in self.layers])
        return loss, cost

    def compute_accuracy(self, X, Y):
        N = X.shape[1]
        P = self.evaluate_classifier(X)
        y_pred = np.argmax(P, axis=0)
        y = np.argmax(Y, axis=0)
        return (y == y_pred).sum() / N

    def evaluate_classifier(self, X):
        next_input = X.copy()
        for layer in self.layers:
            next_input = layer.evaluate_layer(next_input)
        return next_input

    def backward_pass(self, Y, P, update_params=True):
        N = Y.shape[1]
        G = -(Y-P)
        gradients = []
        for layer in reversed(self.layers):
            grad_W = G @ layer.last_input.T / N + 2*self.lam*layer.W
            grad_b = np.sum(G, axis=1) / N
            grad_b = grad_b[:, None]
            G = layer.W.T @ G
            mask = np.where(layer.last_input > 0, 1, 0)
            G = np.multiply(G, mask)
            gradients.append({'W': grad_W, 'b': grad_b})
            if update_params:
                layer.update_params(grad_W, grad_b, self.eta)
        gradients.reverse()
        return gradients

    def train(self, X, Y, X_val, Y_val):
        N = X.shape[1]
        cycle = 0
        update_num = 0
        metrics = {'cost': [], 'val_cost':[], 'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        for _ in tqdm.trange(self.n_epochs):
            shuffled_ind = np.random.permutation(X.shape[1])
            X_shuffled = X[:, shuffled_ind]
            Y_shuffled = Y[:, shuffled_ind]
            for i in range(int(N / self.n_batch)):
                cycle = update_num // (self.gd_params.n_s*2)
                self.eta = self.gd_params.next_cyclical_learning_rate(
                    update_num, cycle)
                start = i * self.n_batch
                end = start + self.n_batch
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]
                P = self.evaluate_classifier(X_batch)
                self.backward_pass(Y_batch, P, update_params=True)
                    
                update_num += 1
                if self.gd_params.plot and (update_num-1) % 100 == 0:
                    metrics = self.update_metrics(X_batch, Y_batch, X_val, Y_val, P, metrics)
        if self.gd_params.plot:        
            return metrics

        return self.compute_accuracy(X_val, Y_val)

    def compute_gradient(self, X, Y) -> List[dict]:
        # """ Compute next gradients without updating network's parameters. 

        # Args:
        #     X (np.ndarray): data batch matrix (n_dims, n_samples)
        #     Y (np.ndarray): one-hot-encoding labels batch vector (n_classes, n_samples)

        # Returns:
        #     List: list of gradients of W and b layerwise, [{'W': .., 'b': ..}] 
        # """
        P = self.evaluate_classifier(X)
        return self.backward_pass(Y, P, update_params=False)

    def compute_gradients_num(self, X_batch, Y_batch, h=1e-5):
        """ Numerically computes the gradients of the weight and bias parameters
        Args:
            X_batch (np.ndarray): data batch matrix (n_dims, n_samples)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (n_classes, n_samples)
            h            (float): marginal offset
        Returns:
            grad_W  (np.ndarray): the gradient of the weight parameter
            grad_b  (np.ndarray): the gradient of the bias parameter
        """
        grads = {}
        for j, layer in enumerate(self.layers):
            selfW = layer.W
            selfB = layer.b
            grads['W' + str(j)] = np.zeros(selfW.shape)
            grads['b' + str(j)] = np.zeros(selfB.shape)

            b_try = np.copy(selfB)
            for i in range(selfB.shape[0]):
                layer.b = np.copy(b_try)
                layer.b[i] += h
                c1 = self.compute_cost(X_batch, Y_batch)
                layer.b = np.copy(b_try)
                layer.b[i] -= h
                c2 = self.compute_cost(X_batch, Y_batch)
                grads['b' + str(j)][i] = (c1-c2) / (2*h)
            layer.b = b_try

            W_try = np.copy(selfW)
            for i in np.ndindex(selfW.shape):
                layer.W = np.copy(W_try)
                layer.W[i] += h
                c1 = self.compute_cost(X_batch, Y_batch)
                layer.W = np.copy(W_try)
                layer.W[i] -= h
                c2 = self.compute_cost(X_batch, Y_batch)
                grads['W' + str(j)][i] = (c1-c2) / (2*h)
            layer.W = W_try

        return grads['W0'], grads['b0'], grads['W1'], grads['b1']
