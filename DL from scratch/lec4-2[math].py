import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h) 
def function_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1) # 0에서 ~ 20까지 0.1 단위
y = function_1(x)
#plt.plot(x, y)
#plt.show()

nd = numerical_diff(function_1, x)
plt.plot(x, nd)
plt.plot(x, y)
plt.show()

a = np.array([[1, 2, 3, 4],
          [4, 5, 6, 7],
          [8, 9, 3, 1]])

#print(np.linalg.pinv(a))
    
## 4.3.3 partial differential ##
def fuction_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x):
    return x*x + 4.0**2.0

print(numerical_diff(function_tmp1, 3.0))

def numericalgradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    return grad


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        