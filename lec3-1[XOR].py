import numpy as np

X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
y = np.array([[0], [1], [1], [0]])

X_train = np.insert(X, 2, 1., axis=1)

W1 = np.zeros(shape=[3, 2])
W2 = np.zeros(shape=[3, 1])

W1[0,0] = -0.089
W1[0,1] = 0.098
W1[1,0] = 0.028
W1[1,1] = -0.07
W1[2,0] = 0.092
W1[2,1] = -0.01

W2[0] = 0.056
W2[1] = 0.067
W2[2] = 0.016

net_n1 = np.matmul(X_train, W1)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))

h_n1 = sigmoid(net_n1)
net_n1_bias = np.insert(h_n1, 2, 1., axis=1)

net_n2 = np.matmul(net_n1_bias, W2)
o_n2 = sigmoid(net_n2)

def cost(y_hat):
    return np.mean(- y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat))

learning_rate = 0.001
 