from six.moves import cPickle
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from functions import softmax
from tqdm import tqdm

def loadData(filename, reshape=False, clipping=False):
    
    f = open('Dataset/cifar-10-batches-py/'+filename, 'rb')
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
    ## One hot Encode labels
    Y = to_categorical(y, num_classes=10)
    Y = Y.T

    return X, y, Y

def plotCifar(X, Y):
    fig, axes1 = plt.subplots(5, 5, figsize=(12, 12))
    print(X.shape)
    for j in range(5):
        for k in range(5):
            i = np.random.choice(range(len(X)))
            axes1[j][k].set_axis_off()
            axes1[j][k].imshow(X[i:i+1][0])
            axes1[j][k].set_title(Y[i:i+1])


def EvaluateClassifier(x, W, b):
    z = W @ x +b 
    return softmax(z)

def ComputeCost(X, Y, W , b, _lambda):
    P = EvaluateClassifier(X, W, b)
    l = -np.log(np.sum(np.multiply(Y, P), axis=0))
    r = np.linalg.norm(W)**2
    J = np.sum(l)/X.shape[1] + _lambda*r
    return J

def ComputeAccuracy(X, y, W, b):
    P = EvaluateClassifier(X, W, b)
    y_pred = np.argmax(P, axis=0)
    return accuracy_score(y, y_pred)

def ComputeGradients(X, Y, P, W, _lambda):
    G = -(Y - P)
    nb = X.shape[1]
    grad_W = G @ X.T / nb + 2 * _lambda * W 
    grad_b = np.sum(G, axis=1) / nb

    return grad_W, grad_b

def compare_gradients(ga, gn, eps):
    K, d = ga.shape
    rerr = np.zeros((K, d))
    for i in range(K):
        for j in range(d):
            rerr[i, j] = abs(ga[i, j] - gn[i, j]) / max(eps, abs(ga[i, j]) + abs(gn[i, j]))
    return rerr

def history(X, Y, y, X_val, Y_val, y_val, epoch, W, b, _lambda, train_loss, val_loss):
    
    J_train = ComputeCost(X, Y, W, b, _lambda)
    J_val = ComputeCost(X_val, Y_val, W, b, _lambda)

    train_acc = ComputeAccuracy(X, y, W, b)
    val_acc = ComputeAccuracy(X_val, y_val, W, b)

    print(f'Epoch {epoch}: train_acc={train_acc} | val_acc={val_acc} | train_loss={J_train} | val_loss={J_val}')

    train_loss.append(J_train)
    val_loss.append(J_val)

def minibatchGD(X, Y, y,  X_val, Y_val, y_val, GDparams, W, b, _lambda, seed=42):
    # np.random.seed(seed)
    _, n = X.shape

    train_loss, val_loss =  [], []

    epochs, batch_size, eta = GDparams["epochs"], GDparams["batch_size"], GDparams["eta"]

    history(X, Y, y,  X_val, Y_val, y_val, 0, W, b, _lambda, train_loss, val_loss)

    for epoch in tqdm(range(epochs)):

        # X, Y, y = shuffle(X, Y, y.reshape(-1,1), random_state=epoch)

        for j in range(n//batch_size):
            j_start = j * batch_size
            j_end = (j+1) * batch_size
            X_batch = X[:, j_start:j_end]
            Y_batch = Y[:, j_start:j_end]

            # print(X_batch.shape)

            P_batch = EvaluateClassifier(X_batch, W, b)
            # print(P_batch)
            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, P_batch, W, _lambda)
            # print("W", grad_W, "b", grad_b)

            W -= eta * grad_W
            b -= eta * grad_b.reshape(len(b), 1)

        history(X, Y, y,  X_val, Y_val, y_val, epoch, W, b, _lambda, train_loss, val_loss)

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)

    np.save('History/weights_{epochs}_{batch_size}_{eta}_{_lambda}.npy', W)
    np.save('History/bias_{epochs}_{batch_size}_{eta}_{_lambda}.npy', b)
    np.save('History/train_loss_{epochs}_{batch_size}_{eta}_{_lambda}.npy', train_loss)
    np.save('History/val_loss_{epochs}_{batch_size}_{eta}.npy', val_loss)

    return W, b, train_loss, val_loss

def plot_weights(W, epochs, batch_size, eta, _lambda):
    s_im = {}
    K = W.shape[0]
    for i in range(K):
        im = W[i].reshape((32,32,3))
        s_im[i] = ((im - np.min(im))/ (np.max(im)-np.min(im))).transpose(1, 0, 2)
    
    _, axes1 = plt.subplots(1, K, figsize=(20, 5))
    for k in range(K):
        axes1[k].set_axis_off()
        axes1[k].imshow(s_im[i])

    plt.savefig(f'History/weights_{epochs}_{batch_size}_{eta}_{_lambda}.png')


def plot_loss(train_loss, val_loss, epochs, batch_size, eta, _lambda):
    plt.plot(train_loss, label="Train loss")
    plt.plot(val_loss, label="Validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(f'History/hist_{epochs}_{batch_size}_{eta}_{_lambda}.png')
    plt.show()