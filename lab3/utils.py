from six.moves import cPickle
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

def loadData(filename, reshape=False, clipping=False):
    """ Loads data and creates one hot encoded labels """

    f = open('../Dataset/cifar-10-batches-py/'+filename, 'rb')
    datadict = cPickle.load(f, encoding='latin1')
    f.close()

    X = datadict["data"]
    y = datadict['labels']

    if reshape:
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

    if clipping:
        X = X.astype(np.float32)
        X /= 255.0
        X = X.T

    y = np.array(y)
    # One hot Encoded labels
    Y = to_categorical(y, num_classes=10)
    Y = Y.T

    return X, y, Y


def plotCifar(X, Y):
    """ Util function to plot cifar original images along with their labels """
    fig, axes1 = plt.subplots(5, 5, figsize=(12, 12))
    print(X.shape)
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i+1][0])
            axes1[j][k].set_title(Y[i:i+1])


def batch_normalize(X, mean, std):
    Xc = np.copy(X)
    Xc -= np.outer(mean, np.ones(X.shape[1]))
    Xc /= np.outer(std, np.ones(X.shape[1]))
    return X


def preprocess_data(X_train, y_train, Y_train, X_val, y_val, Y_val, X_test, y_test, Y_test):
    mean_X = np.mean(X_train, axis=1)
    std_X = np.std(X_train, axis=1)

    X_train = batch_normalize(X_train, mean_X, std_X)
    X_val = batch_normalize(X_val, mean_X, std_X)
    X_test = batch_normalize(X_test, mean_X, std_X).copy()
    data = {"X_train": X_train, "y_train": y_train, "Y_train": Y_train,
            "X_val": X_val, "y_val": y_val, "Y_val": Y_val, 
            "X_test": X_test, "y_test": y_test, "Y_test": Y_test}
    return data
