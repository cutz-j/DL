import numpy as np
import pandas as pd

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask= (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

if __name__ == '__main__':
    relu = Relu()
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    relu.backward(relu.forward(x))
    sigmoid = Sigmoid()
    sigmoid.backward(sigmoid.forward(x))
    
    
    
    
    
    
    
    
    
    
    
    
    
