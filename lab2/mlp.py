from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from utils import softmax
from tqdm import tqdm


class Layer():
    def __init__(self, d_in, d_out, W, b, grad_W, grad_b, x):
        self.d_in = d_in
        self.d_out = d_out
        self.W = W
        self.b = b
        self.grad_W = grad_W
        self.grad_b = grad_b
        self.input = input


class MLP():
    def __init__(self, k=2, dims=[3072, 50, 10], lamda=0, seed=42) -> None:
        np.random.seed(seed)
        self.seed = seed
        self.k = k
        self.lamda = lamda
        self.dims = dims
        self.layers = []
        for i in range(k):
            d_in, d_out = self.dims[i], self.dims[i+1]
            self.layers.append(Layer(d_in, d_out, np.random.normal(
                0, 1/np.sqrt(d_in), (d_out, d_in)), np.zeros((d_out, 1)), None, None, None))

        self.train_loss, self.val_loss = [], []
        self.train_cost, self.val_cost = [], []
        self.train_acc, self.val_acc = [], []

    def forwardpass(self, X):
        input = X.copy()
        for layer in self.layers:
            layer.input = input.copy()
            input = np.maximum(
                0, layer.W @ layer.input + layer.b)
        return softmax(input)

    def computeCost(self, X, Y):
        """ Computes the cost function: cross entropy loss + L2 regularization """
        P = self.forwardpass(X)
        loss = -np.log(np.sum(np.multiply(Y, P), axis=0))
        loss = np.sum(loss)/X.shape[1]
        r = np.sum([np.linalg.norm(self.layers[i].W)
                    ** 2 for i in range(self.k)])
        cost = loss + self.lamda * r
        return loss, cost

    def computeGradients(self, X, Y, P):
        G = - (Y - P)
        nb = X.shape[1]

        for layer in reversed(self.layers):
            layer.grad_W = G @ layer.input.T / nb + \
                2 * self.lamda * layer.W
            layer.grad_b = (
                        np.sum(G, axis=1) / nb).reshape(layer.d_out, 1)
            G = layer.W.T @ G
            G = np.multiply(G, np.heaviside(layer.input, 0))

    def updateParameters(self, eta=1e-2):
        for layer in self.layers:
            layer.W -= eta * layer.grad_W
            layer.b -= eta * layer.grad_b

    def computeGradientsNum(self, X, Y, h=1e-5):
        grad_bs, grad_Ws = [], []

        for j, layer in tqdm(enumerate(self.layers)):
            grad_bs.append(np.zeros(layer.d_out))

            b_copy = np.copy(layer.b)
            for i in range(layer.d_out):
                layer.b = np.copy(b_copy)
                layer.b[i] -= h
                _, c1 = self.computeCost(X, Y)
                
                layer.b = np.copy(b_copy)
                layer.b[i] += h
                _, c2 = self.computeCost(X, Y)

                grad_bs[j][i] = (c2 - c1) / (2*h)
            layer.b = b_copy

            grad_Ws.append(
                np.zeros((layer.d_out, layer.d_in)))

            W_copy = np.copy(layer.W) 
            for i in range(layer.d_out):
                for l in range(layer.d_in):
                    layer.W = np.copy(W_copy)
                    layer.W[i, l] -= h
                    _, c1 = self.computeCost(X, Y)

                    layer.W = np.copy(W_copy)
                    layer.W[i, l] += h
                    _, c2 = self.computeCost(X, Y)

                    grad_Ws[j][i, l] = (c2 - c1) / (2*h)
            layer.W = W_copy

        return grad_Ws, grad_bs

    def compareGradients(self, X, Y, eps=1e-10, h=1e-5):
        """ Compares analytical and numerical gradients given a certain epsilon """
        gn_Ws, gn_bs = self.computeGradientsNum(X, Y, h)
        rerr_w, rerr_b = [], []
        aerr_w, aerr_b = [], []

        def _rel_error(x, y, eps): return np.abs(
            x-y)/max(eps, np.abs(x)+np.abs(y))

        def rel_error(g1, g2, eps):
            vfunc = np.vectorize(_rel_error)
            return np.mean(vfunc(g1, g2, eps))

        for i, layer in enumerate(self.layers):
            rerr_w.append(rel_error(layer.grad_W, gn_Ws[i], eps))
            rerr_b.append(rel_error(layer.grad_b, gn_bs[i], eps))
            aerr_w.append(np.mean(abs(layer.grad_W - gn_Ws[i])))
            aerr_b.append(np.mean(abs(layer.grad_b - gn_bs[i])))

        return rerr_w, rerr_b, aerr_w, aerr_b

    def computeAccuracy(self, X, y):
        """ Computes the prediction accuracy of a given state of the network """
        P = self.forwardpass(X)
        y_pred = np.argmax(P, axis=0)
        return accuracy_score(y, y_pred)

    def minibatchGD(self, data, GDparams, verbose=True, backup=False):
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

                P_batch = self.forwardpass(X_batch)

                self.computeGradients(X_batch, Y_batch, P_batch)

                self.updateParameters(eta)

            self.history(data, epoch, verbose, cyclic=False)

        if backup:
            self.backup(GDparams)

    def cyclicLearning(self, data, GDparams, verbose=True, backup=False):
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

                P_batch = self.forwardpass(X_batch)

                self.computeGradients(X_batch, Y_batch, P_batch)
                self.updateParameters(eta)

                if t % (2*ns/freq) == 0:
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

        t_loss, t_cost = self.computeCost(X, Y)
        v_loss, v_cost = self.computeCost(X_val, Y_val)

        t_acc = self.computeAccuracy(X, y)
        v_acc = self.computeAccuracy(X_val, y_val)

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
    def loadMLP(GDparams, cyclic=True, k=2, dims=[3072, 50, 10], lamda=0, seed=42):
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


class Search():

    def __init__(self, l_min, l_max, n_lambda, p1, params1, p2, params2):
        self.l_min = l_min
        self.l_max = l_max
        self.lambdas = []
        self.n_lambda = n_lambda
        self.p1 = p1
        self.params1 = params1
        self.p2 = p2
        self.params2 = params2
        self.models = {}

    def sampleLambda(self):
        r = self.l_min + (self.l_max - self.l_min) * \
            np.random.rand(self.n_lambda[0])
        self.lambdas = [10**i for i in r]

    def randomSearch(self, data, GDparams):
        self.sampleLambda()
        for t in range(len(self.n_lambda)-1):
            for lmda in self.lambdas:
                if self.params1 and self.params2:
                    self.gridSearch(data, GDparams, lmda)
                else:
                    mlp = MLP(lamda=lmda)
                    mlp.cyclicLearning(data, GDparams, verbose=False, backup=False)
                    self.models.update({mlp.val_acc[-1]: mlp})
            self.updateLambda(n=self.n_lambda[t+1])

        max_key = max(self.models.keys())
        return self.models[max_key]

    def gridSearch(self, data, GDparams, lmda):

        for param1 in self.params1:
            for param2 in self.params2:
                GDparams[self.p1] = param1
                GDparams[self.p2] = param2
                mlp = MLP(lamda=lmda)
                mlp.cyclicLearning(
                    data, GDparams, verbose=False, backup=False)
                self.models.update({mlp.val_acc[-1]: mlp})

    def updateLambda(self, n, _min=1e-2, _max=1e-2):
        key = max(self.models.keys())
        lba = self.models[key].lamda
        l_min = lba - _min
        l_max = lba + _max
        r = l_min + ((l_max - l_min) * np.random.rand(n))
        self.lambdas = [lmbda for lmbda in r]
