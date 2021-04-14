from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def batch_normalize(X, mean, std):
    return (X- mean)/std


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.maximum(0, x)


class Layer():
    def __init__(self, d_in, d_out, W, b, activation):
        self.d_in = d_in
        self.d_out = d_out
        self.W = W
        self.b = b
        self.grad_W = None
        self.grad_b = None
        self.input = None
        self.activation = activation

    def evaluate_layer(self, input, train_mode=True):
        self.input = input.copy()
        return self.activation(self.W @ self.input + self.b)


class BNLayer(Layer):
    def __init__(self, d_in, d_out, W, b, activation, alpha=0.9):
        super().__init__(d_in, d_out, W, b, activation)
        self.scores = None
        self.scores_hat = None
        self.mu = np.zeros((self.d_out, 1))
        self.v = np.zeros((self.d_out, 1))
        self.scale = np.ones((self.d_out, 1))
        self.shift = np.zeros((self.d_out, 1))
        self.alpha = alpha

    def evaluate_layer(self, input, train_mode=True):
        self.input = input.copy()
        self.scores = self.W @ self.input + self.b
        
        if train_mode:
            mu = np.mean(self.scores, axis=1, keepdims=True)
            v = np.var(self.scores, axis=1, ddof=1, keepdims=True)
            
            self.scores_hat = batch_normalize(
                self.scores, mu, np.sqrt(v + np.finfo(float).eps))

            self.mu = self.alpha * self.mu + (1-self.alpha) * mu
            self.v = self.alpha * self.v + (1-self.alpha) * v
        
        else:
            self.scores_hat = batch_normalize(
                self.scores, self.mu, np.sqrt(self.v + np.finfo(float).eps))
        
        return self.activation(np.multiply(self.scale, self.scores_hat) + self.shift)


class MLP():
    def __init__(self, k=2, dims=[3072, 50, 10], lamda=0, seed=42, batch_norm=False, alpha=0.9):
        np.random.seed(seed)
        self.seed = seed
        self.k = k
        self.lamda = lamda
        self.dims = dims
        self.layers = []
        self.batch_norm = batch_norm
        self.alpha = alpha
        self.add_layers()
        self.train_loss, self.val_loss = [], []
        self.train_cost, self.val_cost = [], []
        self.train_acc, self.val_acc = [], []

    def add_layers(self):
        for i in range(self.k):
            d_in, d_out = self.dims[i], self.dims[i+1]
            activation = relu if i < self.k-1 else softmax
            if self.batch_norm and i < self.k-1:
                layer = BNLayer(d_in, d_out, np.random.normal(
                    0, 1/np.sqrt(d_in), (d_out, d_in)), np.zeros((d_out, 1)), activation, alpha=self.alpha)
            else:
                layer = Layer(d_in, d_out, np.random.normal(
                    0, 1/np.sqrt(d_in), (d_out, d_in)), np.zeros((d_out, 1)), activation)

            self.layers.append(layer)

    def forward_pass(self, X, train_mode=True):
        input = X.copy()
        for layer in self.layers:
            input = layer.evaluate_layer(input, train_mode)
        return input

    def compute_cost(self, X, Y, train_mode=True):
        """ Computes the cost function: cross entropy loss + L2 regularization """
        P = self.forward_pass(X, train_mode)
        loss = - np.sum(np.log(np.sum(np.multiply(Y, P), axis=0)))/X.shape[1]
        r = np.sum([np.linalg.norm(layer.W) ** 2 for layer in self.layers])
        cost = loss + self.lamda * r
        return loss, cost

    def compute_gradients(self, X, Y, P):
        G = - (Y - P)
        nb = X.shape[1]

        for layer in reversed(self.layers):
            layer.grad_W = G @ layer.input.T / nb + \
                2 * self.lamda * layer.W
            layer.grad_b = (
                np.sum(G, axis=1) / nb).reshape(layer.d_out, 1)
            G = layer.W.T @ G
            G = np.multiply(G, np.heaviside(layer.input, 0))

    def update_parameters(self, eta=1e-2):
        for layer in self.layers:
            layer.W -= eta * layer.grad_W
            layer.b -= eta * layer.grad_b

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
                _, c1 = self.compute_cost(X_batch, Y_batch)
                layer.b = np.copy(b_try)
                layer.b[i] -= h
                _, c2 = self.compute_cost(X_batch, Y_batch)
                grads['b' + str(j)][i] = (c1-c2) / (2*h)
            layer.b = b_try

            W_try = np.copy(selfW)
            for i in np.ndindex(selfW.shape):
                layer.W = np.copy(W_try)
                layer.W[i] += h
                _, c1 = self.compute_cost(X_batch, Y_batch)
                layer.W = np.copy(W_try)
                layer.W[i] -= h
                _, c2 = self.compute_cost(X_batch, Y_batch)
                grads['W' + str(j)][i] = (c1-c2) / (2*h)
            layer.W = W_try

        return grads

    def compare_gradients(self, X, Y, eps=1e-10, h=1e-5):
        """ Compares analytical and numerical gradients given a certain epsilon """
        gn = self.compute_gradients_num(X, Y, h)
        rerr_w, rerr_b = [], []
        aerr_w, aerr_b = [], []

        def _rel_error(x, y, eps): return np.abs(
            x-y)/max(eps, np.abs(x)+np.abs(y))

        def rel_error(g1, g2, eps):
            vfunc = np.vectorize(_rel_error)
            return np.mean(vfunc(g1, g2, eps))

        for i, layer in enumerate(self.layers):
            rerr_w.append(rel_error(layer.grad_W, gn[f'W{i}'], eps))
            rerr_b.append(rel_error(layer.grad_b, gn[f'b{i}'], eps))
            aerr_w.append(np.mean(abs(layer.grad_W - gn[f'W{i}'])))
            aerr_b.append(np.mean(abs(layer.grad_b - gn[f'b{i}'])))

        return rerr_w, rerr_b, aerr_w, aerr_b

    def compute_accuracy(self, X, y):
        """ Computes the prediction accuracy of a given state of the network """
        P = self.forward_pass(X)
        y_pred = np.argmax(P, axis=0)
        return accuracy_score(y, y_pred)

    def mini_batch_gd(self, data, GDparams, verbose=True, backup=False):
        """ Performas minibatch gradient descent """

        X, Y, y = data["X_train"], data["Y_train"], data["y_train"]

        _, n = X.shape

        epochs, batch_size, eta = GDparams["n_epochs"], GDparams["n_batch"], GDparams["eta"]

        self.history(data, 0, verbose, cyclic=False)

        for epoch in tqdm(range(epochs)):

            X, Y, y = shuffle(X.T, Y.T, y.T, random_state=epoch)
            X, Y, y = X.T, Y.T, y.T

            for j in range(n//batch_size):
                j_start = j * batch_size
                j_end = (j+1) * batch_size
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                P_batch = self.forward_pass(X_batch)

                self.compute_gradients(X_batch, Y_batch, P_batch)

                self.update_parameters(eta)

            self.history(data, epoch, verbose, cyclic=False)

        if backup:
            self.backup(GDparams)

    def cyclic_learning(self, data, GDparams, verbose=True, backup=False):
        """ Performas minibatch gradient descent """
        X, Y, y = data["X_train"], data["Y_train"], data["y_train"]

        _, n = X.shape

        n_cycles, batch_size, eta_min, eta_max, ns, freq = GDparams["n_cycles"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"], GDparams['freq']

        eta = eta_min
        t = 0

        epochs = batch_size * 2 * ns * n_cycles // n

        for epoch in tqdm(range(epochs)):
            X, Y, y = shuffle(X.T, Y.T, y.T, random_state=epoch)
            X, Y, y = X.T, Y.T, y.T
            for j in range(n//batch_size):
                j_start = j * batch_size
                j_end = (j+1) * batch_size
                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                P_batch = self.forward_pass(X_batch)

                self.compute_gradients(X_batch, Y_batch, P_batch)
                self.update_parameters(eta)

                if t % (2*ns//freq) == 0:
                    self.history(data, t, verbose)

                if t <= ns:
                    eta = eta_min + t/ns * (eta_max - eta_min)
                else:
                    eta = eta_max - (t - ns)/ns * (eta_max - eta_min)

                t = (t+1) % (2*ns)
        if backup:
            self.backup_cyclic(GDparams)

    def history(self, data, epoch, verbose=True, cyclic=True):
        """ Creates history of the training """

        X, Y, y, X_val, Y_val, y_val = data["X_train"], data["Y_train"], data[
            "y_train"], data["X_val"], data["Y_val"], data["y_val"]

        t_loss, t_cost = self.compute_cost(X, Y)
        v_loss, v_cost = self.compute_cost(X_val, Y_val)

        t_acc = self.compute_accuracy(X, y)
        v_acc = self.compute_accuracy(X_val, y_val)

        if verbose:
            pref = "Update Step " if cyclic else "Epoch "
            print(
                f'{pref}{epoch}: train_acc={t_acc} | val_acc={v_acc} | train_loss={t_loss} | val_loss={v_loss} | train_cost={t_cost} | val_cost={v_cost}')

        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.train_cost.append(t_cost)
        self.val_cost.append(v_cost)
        self.train_acc.append(t_acc)
        self.val_acc.append(v_acc)

    def backup(self, GDparams):
        """ Saves networks params in order to be able to reuse it """

        epochs, batch_size, eta, exp = GDparams["n_epochs"], GDparams["n_batch"], GDparams["eta"], GDparams["exp"]

        np.save(
            f'History/{exp}_layers_{epochs}_{batch_size}_{eta}_{self.lamda}_{self.seed}.npy', self.layers)

        hist = {"train_loss": self.train_loss, "train_acc": self.train_acc, "train_cost": self.train_cost,
                "val_loss": self.val_loss, "val_acc": self.val_acc, "val_cost": self.val_cost}

        np.save(
            f'History/{exp}_hist_{epochs}_{batch_size}_{eta}_{self.lamda}_{self.seed}.npy', hist)

    def backup_cyclic(self, GDparams):
        """ Saves networks params in order to be able to reuse it for cyclic learning"""

        n_cycles, batch_size, eta_min, eta_max, ns, exp = GDparams["n_cycles"], GDparams[
            "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"], GDparams["exp"]

        np.save(
            f'History/{exp}_layers_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.npy', self.layers)

        hist = {"train_loss": self.train_loss, "train_acc": self.train_acc, "train_cost": self.train_cost,
                "val_loss": self.val_loss, "val_acc": self.val_acc, "val_cost": self.val_cost}

        np.save(
            f'History/{exp}_hist_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.npy', hist)

    def plot_metric(self, GDparams, metric="loss", cyclic=True):
        """ Plots a given metric (loss or accuracy) """

        if cyclic:
            n_cycles, batch_size, eta_min, eta_max, ns = GDparams["n_cycles"], GDparams[
                "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"]
        else:
            epochs, batch_size, eta = GDparams["n_epochs"], GDparams["n_batch"], GDparams["eta"]

        batch_size, exp = GDparams["n_batch"], GDparams['exp']

        if metric == "loss":
            plt.ylim(0, 3)
            plt.plot(self.train_loss, label=f"Train {metric}")
            plt.plot(self.val_loss, label=f"Validation {metric}")
        elif metric == "accuracy":
            plt.ylim(0, 0.8)
            plt.plot(self.train_acc, label=f"Train {metric}")
            plt.plot(self.val_acc, label=f"Validation {metric}")
        else:
            plt.ylim(0, 4)
            plt.plot(self.train_cost, label=f"Train {metric}")
            plt.plot(self.val_cost, label=f"Validation {metric}")

        plt.xlabel("epochs")
        plt.ylabel(metric)
        if cyclic:
            plt.title(f"Monitoring of {metric} during {n_cycles} cycles.")
        else:
            plt.title(f"Monitoring of {metric} during {epochs} epochs.")
        plt.legend()
        if cyclic:
            plt.savefig(
                f'History/{exp}_{metric}_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{self.lamda}_{self.seed}.png')
        else:
            plt.savefig(
                f'History/{exp}_{metric}_{epochs}_{batch_size}_{eta}_{self.lamda}_{self.seed}.png')
        plt.show()

    @staticmethod
    def load_mlp(GDparams, cyclic=True, k=2, dims=[3072, 50, 10], lamda=0, seed=42):
        mlp = MLP(k, dims, lamda, seed)
        if cyclic:

            n_cycles, batch_size, eta_min, eta_max, ns, exp = GDparams["n_cycles"], GDparams[
                "n_batch"], GDparams["eta_min"], GDparams["eta_max"], GDparams["ns"], GDparams["exp"]
            layers = np.load(
                f'History/{exp}_layers_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)
            hist = np.load(
                f'History/{exp}_hist_{n_cycles}_{batch_size}_{eta_min}_{eta_max}_{ns}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)
        else:

            epochs, batch_size, eta, exp = GDparams["n_epochs"], GDparams[
                "n_batch"], GDparams["eta"], GDparams["exp"]

            layers = np.load(
                f'History/{exp}_layers_{epochs}_{batch_size}_{eta}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)

            hist = np.load(
                f'History/{exp}_hist_{epochs}_{batch_size}_{eta}_{mlp.lamda}_{mlp.seed}.npy', allow_pickle=True)

        mlp.layers = layers

        mlp.train_acc = hist.item()['train_acc']
        mlp.train_loss = hist.item()["train_loss"]
        mlp.train_cost = hist.item()["train_cost"]
        mlp.val_acc = hist.item()['val_acc']
        mlp.val_loss = hist.item()["val_loss"]
        mlp.val_cost = hist.item()["val_cost"]

        return mlp
