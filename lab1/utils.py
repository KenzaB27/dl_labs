from six.moves import cPickle
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename, reshape=False, normalize=False):
    
    f = open('Dataset/cifar-10-batches-py/'+filename, 'rb')
    datadict = cPickle.load(f, encoding='latin1')
    f.close()

    X = datadict["data"]
    y = datadict['labels']
    
    if reshape:
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    
    if normalize:
        X = X.astype(np.float32)
        X /= 255.0

    y = np.array(y)
    ## One hot Encode labels
    Y = to_categorical(y, num_classes=10)

    return X, y, Y

def plot_cifar(X, Y):
    fig, axes1 = plt.subplots(5, 5, figsize=(12, 12))

    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i+1][0])
            axes1[j][k].set_title(Y[i:i+1])
