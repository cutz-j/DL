import numpy as np
import tensorflow as tf

a=np.array([0.3, 2.9, 4.0])

def softmax_norm(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    return exp_a/sum_exp_a

#print(softmax(a))

a=np.array([1010, 1000, 990])

def softmax(a):
    a_max=(np.max(a))
    exp_a=np.exp(a-a_max)
    sum_exp_a=np.sum(exp_a)
    return exp_a/sum_exp_a

#print(softmax_sol(a))