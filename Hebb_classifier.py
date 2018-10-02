import numpy as np
from tensorflow import keras
from sklearn import datasets

def load_data(datasetname):
    if datasetname == 'mnist':
        return keras.datasets.mnist.load_data()
    if datasetname == 'iris':
        iris = datasets.load_iris()
        X = iris['data']
        y = iris['target']
        X_train = np.concatenate((X[0:40], X[50:90], X[100:140]), axis=0)
        y_train = np.concatenate((y[0:40], y[50:90], y[100:140]))
        X_test = np.concatenate((X[40:50], X[90:100], X[140:150]), axis=0)
        y_test = np.concatenate((y[40:50], y[90:100], y[140:150]))
        return ((X_train, y_train), (X_test, y_test))

## preprocess
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def iris_preprocess(x):
    x1 = x/np.max(x, axis=0)
    x2 = 1 - x1
    x3 = 1 - np.abs(x1-0.5)
    result = np.concatenate((x1, x2, x3), axis=1)
    return result

## Hebb's rule
def hebian_update(x, y, W):
    x = x.reshape(x.shape[0],1)
    y = y.reshape(1, y.shape[0])
    sign = 0.01*(np.dot(x,y) - np.dot(1-x,y))
    #delta = 0.01*sign*(1 - sigmoid(W*sign))
    delta_W = 0.01*np.dot(x - np.dot(W,y.transpose()), y)
    W += delta_W
    W /= np.linalg.norm(W, axis=0)

def hebian_fit(images, labels, W):
    for temp in range(5):
        for i in range(images.shape[0]):
            hebian_update(images[i], labels[i], W)

## train 
train, test = load_data('iris')
x, y = train
ground_truth = y
y = keras.utils.to_categorical(y)

x = iris_preprocess(x)

W = np.random.rand(x.shape[1], y.shape[1])
hebian_fit(x, y, W)

pred_train = np.argmax(sigmoid(np.dot(x, W)), axis=1)
acc = np.sum(pred_train == ground_truth)/ground_truth.shape[0]

## Test
x_test, y_test = test
y_test = keras.utils.to_categorical(y_test)

feat_test = iris_prepocess(x_test)
pred = np.argmax(np.dot(feat_test, W),1)
