import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import math
# import numpy as np

def evaluate_hw1():
    batch_size = 256
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transform,
                          )
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
    NN = torch.load("model.pkl")
    n_sampels = 0
    n_correct = 0
    for i,(x, y) in enumerate(test_loader):
        x = x.view(-1,28*28)
        x = x.float()
        y = y.view(-1,1)
        out = NN.forward(x)
        n_sampels += len(y)
        n_correct += score_function(out , y)
    print(n_correct/n_sampels)
    
def score_function(out , trues):
    out = torch.argmax(out, dim=1)
    trues = trues.view((-1))
    test_correct = out.eq(trues).sum().item()
    return test_correct

class Neural_Network:
    def __init__(self, input_size=28*28, output_size=10, hidden_size=50):
        # parameters
        self.inputSize = input_size 
        self.outputSize = output_size
        self.hiddenSize = hidden_size
        
        # weights
        self.W1 = (torch.randn(self.inputSize, self.hiddenSize)/10).float()
        self.b1 = (torch.zeros(self.hiddenSize)).float()
        
        self.W2 = (torch.randn(self.hiddenSize, self.outputSize)).float()
        self.b2 = (torch.zeros(self.outputSize)).float()
        
    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = tanh(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return softmax(self.z2) # need to change to softmax

    
    def backward(self, X, y, y_hat, lr=.1):

        m = y.shape[0]
        dl_dz2 = y_hat
        dl_dz2[range(m),y] -= 1
        dl_dz2 = dl_dz2/m
        dl_dh = torch.matmul(dl_dz2.float(), torch.t(self.W2).float())
        dl_dz1 = dl_dh * tanhPrime(self.h)
        
        self.W1 -= lr*torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr*torch.matmul(torch.t(dl_dz1), torch.ones(m))
        self.W2 -= lr*torch.matmul(torch.t(self.h).float(), dl_dz2.float())
        self.b2 -= lr*torch.matmul(torch.t(dl_dz2).float(), torch.ones(m).float())
    
        
    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o, lr=0.01)

def softmax(X): # need to change the function
    x = torch.tensor(X, dtype=torch.float64)
    e_x = torch.exp(x)
    results = e_x / e_x.sum(axis=0)
    return results

def softmaxPrime(s):
    length = len(s)
    grad = torch.tensor((length, length))
    for i in range(length):
        for j in range(length):
            if i == j:
                grad[i][j] = s[i]*(1-s[i])
            else:
                grad[i][j] = -s[i]*s[j]
    return grad


def ReLu(t):
    zero_matrix = torch.zeros((t.shape))
    temp = torch.where(t > 0, t, zero_matrix)
    return temp

def ReLuPrime(t):
    zero_matrix = torch.zeros((t.shape))
    one_matrix = torch.ones((t.shape))
    temp = torch.where(t > 0, one_matrix, zero_matrix)
    return temp      

def tanh(t):
    return torch.div(torch.exp(t) - torch.exp(-t), torch.exp(t) + torch.exp(-t))

def tanhPrime(t):
    return 1 - t*t

def sigmoid(s):
    return 1 / (1 + torch.exp(-s))

def sigmoidPrime(s):
    return s * (1 - s)
        
        
        
if __name__ == "__main__":
    evaluate_hw1()