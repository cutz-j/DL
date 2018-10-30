import numpy as np
import sys, os
os.chdir("d:/data/prac/")
from mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {} # 가중치 초기화
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = self.sigmoid(a2)
        y = self.softmax(z2)
        return y
    
    def sigmoid(self, z):
        return 1. / (1 + np.exp(-z))
    
    def softmax(self, y):
        y = -np.log((1. / y) - 1)
        return np.exp(y) / (np.sum(np.exp(y), axis=1)).reshape(len(y), 1)
    
    def cost(self, x, t):
        y = self.predict(x)
        return self.cross_entropy_error(y, t)
    
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
        if t.size == y.size:
            t = t.argmax(axis=1)
                 
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accruacy = np.sum(y == t) / float(x.shape[0])
        return accruacy
    
    def gradient(self, x, t):
        cost_W = lambda W: self.cost(x, t)
        grads = {}
        grads['W1'] = self.numerical_gradient(cost_W, self.params['W1'])
        grads['b1'] = self.numerical_gradient(cost_W, self.params['b1'])
        grads['W2'] = self.numerical_gradient(cost_W, self.params['W2'])
        grads['b2'] = self.numerical_gradient(cost_W, self.params['b2'])
        return grads
    
    def numerical_gradient(self, f, x):
        h = 1e-4 # 0.0001
        grad = np.zeros_like(x)
        
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(x) # f(x+h)
            x[idx] = tmp_val - h 
            fxh2 = f(x) # f(x-h)
            grad[idx] = (fxh1 - fxh2) / (2*h)
            
            x[idx] = tmp_val # 값 복원
            it.iternext()   
            
        return grad
    
    
if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)
    train_loss_list = []
    
    # hyper parameter #
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1
    
    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    
    for i in range(iters_num):  # 총 반복
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        y_batch = y_train[batch_mask]
        
        # 기울기 계산 #
        grad = network.gradient(x_batch, y_batch) # cost
        
        # W 갱신 #
        for key in ('W1', 'b1', 'W2', 'b2'): # 가중치 갱신
            network.params[key] -= learning_rate * grad[key]
            
        cost = network.cost(x_batch, y_batch)
        train_loss_list.append(cost)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        
    
    
        
        
        
        
        
        