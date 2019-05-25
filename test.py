import numpy as np

arr2 = np.array([2,2,4,4])

np.mean(arr2)

def sigmoid(z):
    return np.exp(z) / 1 + np.exp(z)

print(sigmoid(0.6))

a = np.dot([1,1], [0,1])
