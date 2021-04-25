import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from tqdm import tqdm


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x - np.max(x, axis=0)) / \
        np.exp(x - np.max(x, axis=0)).sum(axis=0)



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

    def get_one_hot(self, ix, keepdims=True):
        if keepdims:
            x = np.zeros((self.vocab_len, 1))
        else:
            x = np.zeros(self.vocab_len)
        x[ix] = 1
        return x

    def one_hot_encode_X(self, X, keepdims=True):
        X_ind = np.array([self.char_to_ind[x] for x in X])
        _1hot = np.array(
            [self.get_one_hot(ix=x, keepdims=keepdims) for x in X_ind])
        return X_ind, _1hot

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

        text = ""
        
        if onehot:
            ht, xt = h0, self.data.get_one_hot(i0, keepdims=True)
        else:
            ht, xt = h0, i0
        
        for _ in range(n):
            _, ht, _, pt = self.evaluate_vanilla_rnn(ht, xt)
            # it = self.sample_character(pt)
            it = np.random.choice(range(self.K), p=pt.flat)
            xt = self.data.get_one_hot(it)
            text += self.data.ind_to_char[it]

        return text

    def compute_gradients_num(self, X, y, hprev, h, num_comps=20):
        rnn_params = {"W": self.W, "U": self.U,
                      "V": self.V, "b": self.b, "c": self.c}
        num_grads = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U),
                     "V": np.zeros_like(self.V), "b": np.zeros_like(self.b),
                     "c": np.zeros_like(self.c)}

        for key in rnn_params:
            for i in range(num_comps):
                old_par = rnn_params[key].flat[i]  # store old parameter
                rnn_params[key].flat[i] = old_par + h
                l1 = self.forward_pass(hprev, X, y)
                rnn_params[key].flat[i] = old_par - h
                l2 = self.forward_pass(hprev, X, y)
                # reset parameter to old value
                rnn_params[key].flat[i] = old_par
                num_grads[key].flat[i] = (l1 - l2) / (2*h)

        return num_grads

    def check_gradients(self, X, y, hprev, num_comps=20):

        self.back_propagation(hprev, X, y)
        grads_ana = {"W": self.grads.W, "U": self.grads.U,
                     "V": self.grads.V, "b": self.grads.b, "c": self.grads.c}
        grads_num = self.compute_gradients_num(X, y, hprev, 1e-5)

        print("Gradient checks:")
        for grad in grads_ana:
            num   = abs(grads_ana[grad].flat[:num_comps] -
                    grads_num[grad].flat[:num_comps])
            denom = np.asarray([max(abs(a), abs(b)) + 1e-10 for a,b in
                zip(grads_ana[grad].flat[:num_comps],
                    grads_num[grad].flat[:num_comps])
            ])
            max_rel_error = max(num / denom)

            print("The maximum relative error for the %s gradient is: %e." %
                    (grad, max_rel_error))
        print()

    def forward_pass(self, h, X, y):
        seq_length = len(X)
        loss = 0
        self.h[-1] = np.copy(h)
        for t in range(seq_length):
            self.a[t], self.h[t], self.o[t], self.p[t] = self.evaluate_vanilla_rnn(
                self.h[t-1], X[t])
            loss += -np.log(self.p[t][y[t]][0])
        return loss

    def backward_pass(self, X, y):

        seq_length = len(X) 

        grads_a = np.zeros((self.m, 1))
        grads_o = np.zeros((self.K, 1))
        grads_h = np.zeros((self.m, 1))
        grads_h_next = np.zeros((self.m, 1))

        for t in reversed(range(seq_length)):
            grads_o = np.copy(self.p[t])
            grads_o[y[t]] -= 1

            self.grads.V += grads_o @ self.h[t].T
            self.grads.c += grads_o

            grads_h = self.V.T @ grads_o + grads_h_next
            grads_a = np.multiply(grads_h, (1-np.square(self.h[t])))

            self.grads.U += grads_a @ X[t].T
            self.grads.W += grads_a @ self.h[t-1].T
            self.grads.b += grads_a

            grads_h_next = self.W.T @ grads_a

    def back_propagation(self, h0, X, y):
        # reset gradients
        self.grads = Grads(self.m, self.K)
        seq_length = len(X)
        loss = self.forward_pass(h0, X, y)
        self.backward_pass(X, y)
        self.grads.clip_gradients()
        return loss, self.h[seq_length-1]

    def ada_grad(self, eta):

        grads = {"W": self.grads.W, "U": self.grads.U,
                "V": self.grads.V, "b": self.grads.b, "c": self.grads.c}

        rnn = {"W": self.W, "U": self.U, "V": self.V, "b": self.b, "c": self.c}

        mem = {"W": self.mem.W, "U": self.mem.U,
                "V": self.mem.V, "b": self.mem.b, "c": self.mem.c}

        for param in rnn:
            mem[param] += grads[param] ** 2
            rnn[param] -= eta / np.sqrt(mem[param] + np.finfo(np.float64).eps) * grads[param]

    def train_rnn(self, epochs=20, n=200, eta=.1, freq_syn=500, freq_loss=100, verbose=True, backup=True):
        
        data_ind, data_1hot = self.data.one_hot_encode_X(self.data.book_data)

        history_loss, smooth_loss, prev_loss, syn_text, step = [], 0, 200, {}, 0
        s = 0
        for epoch in tqdm(range(epochs)):
            hprev = np.zeros((self.m, 1))
            for e in range(0, len(self.data.book_data) - self.seq_length - 1, self.seq_length):
                X = data_1hot[e: e+self.seq_length]
                Y = data_ind[e+1: e+1+self.seq_length]
                loss, hprev = self.back_propagation(hprev, X, Y)

                self.ada_grad(eta)

                if step == 0 and epoch == 0:
                    smooth_loss = loss
                smooth_loss = .999 * smooth_loss + .001 * loss

                if step % freq_loss == 0:
                    history_loss.append(smooth_loss)
                    if verbose:
                        print(
                            f"Iter={step} | smooth loss={smooth_loss}")

                if step % freq_syn == 0:
                    syn_text[step] = {}
                    syn_text[step] ['loss'] = smooth_loss
                    syn_text[step]['text'] = self.synthesize_text(
                        hprev, X[0], n, onehot=False)
                    if verbose:
                        print(f"Synthetized text | {syn_text[step]['text']}")
                    
                if smooth_loss < 40:
                    if smooth_loss < prev_loss:
                        rnn_params = {"W": self.W.copy(), "V": self.V.copy(),
                                  "U": self.U.copy(), "b": self.b.copy(), "c": self.c.copy()}
                    prev_loss = smooth_loss
                    s = step

                step += 1

        if verbose:
            plt.figure()
            plt.plot(history_loss)
            plt.show()

        np.save(f"History/params_{s}_{prev_loss}.npy", rnn_params)
        return syn_text

    @staticmethod
    def load_rnn(filename):
        params = np.load(filename, allow_pickle=True).item()
        rnn = RNN()
        rnn_params = {"W": rnn.W, "V": rnn.V,
                       "U": rnn.U, "b": rnn.b, "c": rnn.c}
        for p in params:
            rnn_params[p] = params[p].copy()
        return rnn
