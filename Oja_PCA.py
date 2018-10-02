import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


## load data
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

## init weight 
W_raw = np.random.rand(4,2)
W = W_raw/np.linalg.norm(W_raw, axis=0) ## normalize


## define Oja update function
def Oja_update(x, W):
    y = np.dot(x, W)
    delta_W = 0.01*np.dot((x - np.dot(W,y)).reshape(4,1), y.reshape(1,2))
    W += delta_W
    W = W/np.linalg.norm(W, axis=0) ## normalize

## train and plot result over time
fig, axis = plt.subplots(3, 3)
fig.set_size_inches(15,10)
for count in range(18):
    if count % 2 == 0:
        data_PCA = np.dot(X, W)
        c = count//2
        axis[c // 3, c % 3].plot(data_PCA[:50,0], data_PCA[:50,1], 'ro')
        axis[c // 3, c % 3].plot(data_PCA[50:100,0], data_PCA[50:100,1], 'bo')
        axis[c // 3, c % 3].plot(data_PCA[100:150,0], data_PCA[100:150,1], 'go')
    for i in range(150):
        Oja_update(X[i], W)

## compare with orginal PCA of sklearn.
t = PCA(n_components=2).fit_transform(X)
plt.plot(t[:50,0], t[:50,1], 'ro')
plt.plot(t[50:100,0], t[50:100,1], 'bo')
plt.plot(t[100:150,0], t[100:150,1], 'go')
plt.show()
