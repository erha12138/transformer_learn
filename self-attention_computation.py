import numpy as np
import torch
import torch.nn.functional as F

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1).reshape(-1, 1))
    return e_x / np.sum(e_x, axis=1).reshape(-1, 1)


X = np.array([[1.0,0.0,1.0],     # 序列长度为4，每个序列的embedding维度为3
              [0.0,1.0,0.0],
              [1.0,1.0,0.0],
              [0.0,0.0,2.0]])

W_q = np.array([[1.0,0.0,0.0],   # 线性变换矩阵
                [0.0,2.0,0.0],
                [0.0,0.0,0.5]])
W_k = np.array([[1.0,0.0,0.0],
                [0.0,1.0,0.0],
                [0.0,0.0,1.0]])
W_v = np.array([[1.0,0.0,0.0],
                [0.0,0.5,0.0],
                [0.0,0.0,2.0]])

alpha = X @ W_q @ (X @ W_k).T

V = X @ W_v

AlPHA_prime = softmax(alpha)

Attention_socre = AlPHA_prime @ V

 