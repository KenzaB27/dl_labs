import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict

def load_data(filename):
    
    book_data = open(filename, 'r', encoding='utf8').read()
    book_chars = list(set(book_data))

    data = {"book_data": book_data, "book_chars": book_chars,
            "vocab_len": len(book_chars), "char_to_ind": OrderedDict(
                (char, ix) for ix, char in enumerate(book_chars)),
            "ind_to_char": OrderedDict((ix, char) for ix, char in
                                       enumerate(book_chars))}

    return data

class RNN():
    def __init__(self, m=100, vocab_len=80, eta=.1, seq_length=25, sig=.01, seed=42):
        np.random.seed(seed)
        self.seed = seed
        self.m = m
        self.K = vocab_len
        self.eta = eta
        self.seq_length = seq_length 
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.random.normal(0, sig, size=(self.m, self.K))
        self.W = np.random.normal(0, sig, size=(self.m, self.m))
        self.V = np.random.normal(0, sig, size=(self.K, self.m))
