import numpy as np
import matplotlib.pyplot as plt

def step(x):
    y=x>0
    return y.astype(np.float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def identity(x):
    return x

a=np.array([-0.1, 0, 0.5, 0.2])
#print(step(a))

x=np.arange(-5,5, dtype=np.float)
#y=step(x)
y=sigmoid(x)

#plt.plot(x,y)
#plt.ylim(-0.1, 1.1)
#plt.show()

''' 3층 신경망 '''



def hypothesis(X, W, B):
    return np.dot(X, W)+B

def forward():
    X=np.array([1.0, 0.5])
    W1=np.array([[0.1,0.3,0.5], [0.2, 0.4, 0.6]])
    B1=np.array([0.1, 0.2, 0.3])
    W2=np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2=np.array([0.1, 0.2])
    W3=np.array([[0.1,0.3],[0.2,0.4]])
    B3=np.array([0.1,0.2])
    
    y=identity(hypothesis(sigmoid(hypothesis(sigmoid(hypothesis(X, W1, B1)), W2, B2)), W3, B3))
    return y

print(forward())












