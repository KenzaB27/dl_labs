from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import namedtuple

Model = namedtuple('Model', ['weights', 'biases'])

class MLP():
    def __init__(self, k=2, dims=[3072, 50, 10], lamda=0.1) -> None:
        self.k = k
        self.lamda = lamda
        self.dims = dims
        self.model = Model([], [])
        for l in range(k):
            d_in, d_out = self.dims[l], self.dims[l+1]
            self.model.weights.append(
                np.random.normal(0, 1/np.sqrt(d_out), (d_in, d_out)))
            self.model.biases.append(np.zeros((d_out)))

