import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from tqdm import tqdm


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class TextData():
    def __init__(self, filename):
        self.book_data = None
        self.book_chars = None
        self.vocab_len = None 
        self.char_to_ind = None
        self.ind_to_char = None

        self.load_data(filename)

    def load_data(self, filename):

        self.book_data = open(filename, 'r', encoding='utf8').read()
        self.book_chars = np.array(list(set(self.book_data)))
        self.vocab_len = len(self.book_chars)
        self.char_to_ind = OrderedDict(
            (char, ix) for ix, char in enumerate(self.book_chars))
        self.ind_to_char = OrderedDict((ix, char) for ix, char in
                                       enumerate(self.book_chars))

    @ staticmethod
    def get_one_hot(ix, dim, keepdims=False):
        if keepdims:
            x = np.zeros((dim, 1))
        else:
            x = np.zeros(dim)
        x[ix] = 1
        return x

    def one_hot_encode_X(self, X, keepdims=True):
        X_ind = [self.char_to_ind[x] for x in X]
        return np.array([self.get_one_hot(x, self.vocab_len, keepdims=keepdims) for x in X_ind])

class Grads():
    def __init__(self, m=100, K=25):
        self.m, self.K = m, K
        self.U = np.zeros((self.m, self.K))
        self.W = np.zeros((self.m, self.m))
        self.V = np.zeros((self.K, self.m))
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))

    def clip_gradients(self, _min=-5, _max=5):
        self.U = np.clip(self.U, _min, _max)
        self.W = np.clip(self.W, _min, _max)
        self.V = np.clip(self.V, _min, _max)
        self.b = np.clip(self.b, _min, _max)
        self.c = np.clip(self.c, _min, _max)


class RNN():
    def __init__(self, filename='../Dataset/goblet_book.txt', m=100, seq_length=25, sig=.01, seed=42):
        np.random.seed(seed)
        self.seed = seed
        
        self.data = TextData(filename)

        # dimensionality of the hidden state
        self.m = m                      
        self.K = self.data.vocab_len    
        # the length of the input sequence used during training
        self.seq_length = seq_length

        self.U = np.random.normal(0, sig, size=(self.m, self.K))
        self.W = np.random.normal(0, sig, size=(self.m, self.m))
        self.V = np.random.normal(0, sig, size=(self.K, self.m))
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))


        self.grads = Grads(self.m, self.K)
        self.mem = Grads(m=self.m, K=self.K)

        self.a, self.h, self.o, self.p = {}, {}, {}, {}

    @ staticmethod
    def sample_character(p):
        cp = np.cumsum(p)
        a = np.random.rand(1)
        ixs = np.where(cp-a > 0)
        return ixs[0][0]

    def evaluate_vanilla_rnn(self, h, x):
        a = self.W @ h + self.U @ x + self.b
        h = np.tanh(a)
        o = self.V @ h + self.c
        p = softmax(o)
        return a, h, o, p

    def synthesize_text(self, h0, i0, n, onehot=False):
        text, Y = "", []
        if onehot:
            ht, xt = h0, self.data.get_one_hot(i0, self.K, keepdims=True)
        else:
            ht, xt = h0, i0
        for _ in range(n):
            _, ht, _, pt = self.evaluate_vanilla_rnn(ht, xt)
            it = self.sample_character(pt)
            # it = np.random.choice(range(self.K), p=pt.flat)
            xt = self.data.get_one_hot(it, self.K, keepdims=True)
            Y.append(xt)
            text += self.data.ind_to_char[it]
        Y = np.array(Y)
        return Y, text

    def forward_pass(self, h, X, Y):
        loss = 0
        self.h[-1] = np.copy(h)
        for t in range(self.seq_length):
            self.a[t], self.h[t], self.o[t], self.p[t] = self.evaluate_vanilla_rnn(
                self.h[t-1], X[t])
            loss += -np.log(np.sum(np.multiply(Y[t], self.p[t]), axis=0))[0]
        return loss

    def backward_pass(self, X, Y):

        grads_a = np.zeros((self.m, 1))
        grads_o = np.zeros((self.K, 1))
        grads_h = np.zeros((self.m, 1))
        grads_h_next = np.zeros((self.m, 1))

        for t in reversed(range(self.seq_length)):
            grads_o = self.p[t] - Y[t]

            self.grads.V += grads_o @ self.h[t].T
            self.grads.c += grads_o

            grads_h = self.V.T @ grads_o + grads_h_next
            grads_a = np.multiply(grads_h, (1-np.square(self.h[t])))

            self.grads.U += grads_a @ X[t].T
            self.grads.W += grads_a @ self.h[t-1].T
            self.grads.b += grads_a

            grads_h_next = self.W.T @ grads_a

    def back_propagation(self, h0, X, Y):
        loss = self.forward_pass(h0, X, Y)
        self.backward_pass(X, Y)
        self.grads.clip_gradients()
        return loss, self.h[self.seq_length-1]

    def ada_grad(self, eta):

        self.mem.U += self.grads.U ** 2
        self.mem.V += self.grads.V ** 2
        self.mem.W += self.grads.W ** 2
        self.mem.b += self.grads.b ** 2
        self.mem.c += self.grads.c ** 2

        self.U -= eta / np.sqrt(self.mem.U +
                                np.finfo(np.float64).eps) * self.grads.U
        self.V -= eta / np.sqrt(self.mem.V +
                                np.finfo(np.float64).eps) * self.grads.V
        self.W -= eta / np.sqrt(self.mem.W +
                                np.finfo(np.float64).eps) * self.grads.W
        self.b -= eta / np.sqrt(self.mem.b +
                                np.finfo(np.float64).eps) * self.grads.b
        self.c -= eta / np.sqrt(self.mem.c +
                                np.finfo(np.float64).eps) * self.grads.c

    def train_rnn(self, epochs=20, n=200, eta=.1, freq_syn=500, freq_loss=100, verbose=True, backup=True):
        
        data_1hot = self.data.one_hot_encode_X(self.data.book_data)

        history_loss = []
        smooth_loss = 0
        syn_text = {}

        for epoch in tqdm(range(epochs)):
            hprev = np.zeros((self.m, 1))
            for e in range(0, len(self.data.book_data)-1, self.seq_length):
    
                X = data_1hot[e: e+self.seq_length]
                Y = data_1hot[e+1: e+1+self.seq_length]

                if e % freq_syn == 0:
                    syn_text[(epoch+1)*e] = {}
                    syn_text['loss'] = smooth_loss
                    _, syn_text['text'] = self.synthesize_text(
                        hprev, X[0], n, onehot=False)
                    if verbose:
                        print(
                            f"Iter={(epoch+1)*e} | smooth loss={smooth_loss}")
                        print(f"Synthetized text | {syn_text['text']}")

                loss, hprev = self.back_propagation(hprev, X, Y)

                self.ada_grad(eta)

                smooth_loss = .999 * smooth_loss + .001 * loss

                if e % freq_loss == 0:
                    history_loss.append(smooth_loss)
                    if verbose:
                        print(
                            f"Iter={(epoch+1)*e} | smooth loss={smooth_loss}")

        if backup:
            np.save(f"History/rnn_{epoch}_{eta}.npy", self)
            np.save(f"History/text_{epoch}_{eta}.npy", syn_text)
