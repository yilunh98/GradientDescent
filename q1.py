import matplotlib.pyplot as plt
import numpy as np
import math
import random

###############Batch Gradient Descent################# 
def BGD(x,y,yita,exp):
    m = len(x)
    x_b = np.c_[np.ones((m, 1)), x]
    w = np.random.randn(2)

    n=0
    while True:
        n = n+1
        grad = x_b.T.dot(x_b.dot(w)-y)
        w = w - yita*grad  
        error = np.dot(x_b.dot(w)-y,x_b.dot(w)-y)/m
        if error<exp: 
            print('%dth iterations for BGD'%n)
            break

    return w 
        
###############Stochastic Gradient Descent################# 
def SGD(x,y,yita,exp):
    m = len(x)
    x_b = np.c_[np.ones((m, 1)), x]
    w = np.random.randn(2)

    n=0
    while True:
        n = n+1
        #i = random.randint(0,m-1)
        for i in range(m): 
            grad = (x_b[i].dot(w)-y[i])*x_b[i]
            w = w - yita*grad 
        error = np.dot(x_b.dot(w)-y,x_b.dot(w)-y)/m
        if error<exp: 
            print('%dth iterations for SGD'%n)
            break
    return w

###############Regression train & test################# 
def train_regre(M,x,y):
    m = len(x)
    order = np.arange(M+1)
    order = order[:, np.newaxis]
    exp = np.tile(order, [1,m])
    Phit = np.power(x,exp)
    Phi = np.transpose(Phit)
    A = Phit@Phi
    b = Phit@y
    w = np.linalg.solve(A,b)
    Pw = Phi@w
    error = 1/2*Pw.dot(Pw)-y.dot(Pw)+1/2*y.dot(y)
    rms = np.sqrt(2*error/m)
    return [w,rms]
   
def test_regre(M,x,y,w):
    m = len(x)
    order = np.arange(M+1)
    order = order[:, np.newaxis]
    exp = np.tile(order, [1,m])
    Phit = np.power(x,exp)
    Phi = np.transpose(Phit)
    
    Pw = Phi@w
    error = 1/2*Pw.dot(Pw)-y.dot(Pw)+1/2*y.dot(y)
    rms = np.sqrt(2*error/m)
    return rms
   
###############Regularized Regression train & test################# 
def train_reg(M,x,y,lamda):
    m = len(x)
    order = np.arange(M+1)
    order = order[:, np.newaxis]
    exp = np.tile(order, [1,m])
    Phit = np.power(x,exp)
    Phi = np.transpose(Phit)
    A = Phit@Phi+lamda*np.identity(M+1)
    b = Phit@y
    w = np.linalg.solve(A,b)
    Pw = Phi@w
    error = 1/2*Pw.dot(Pw)-y.dot(Pw)+1/2*y.dot(y)
    rms = np.sqrt(2*error/m)
    return [w,rms]   

def test_reg(M,x,y,w):
    m = len(x)
    order = np.arange(M+1)
    order = order[:, np.newaxis]
    exp = np.tile(order, [1,m])
    Phit = np.power(x,exp)
    Phi = np.transpose(Phit)
    
    Pw = Phi@w
    error = 1/2*Pw.dot(Pw)-y.dot(Pw)+1/2*y.dot(y)
    rms = np.sqrt(2*error/m)
    return rms


if __name__ == '__main__':

###############load data#################
    xTrain = np.load('q1xTrain.npy')
    yTrain = np.load('q1yTrain.npy')
    xTest = np.load('q1xTest.npy')
    yTest = np.load('q1yTest.npy')
 
###############Batch Gradient Descent################# 
    yita = 1e-3
    exp = 0.2
    [a1,b1] = BGD(xTrain,yTrain,yita,exp)
    X1 = range(10)
    Y1 = [(a1*s+b1) for s in X1]
    print('Batch Gradient Descent: y=%fX+%f'%(a1,b1))
 
###############Stochastic Gradient Descent################# 
    [a2,b2] = SGD(xTrain,yTrain,yita,exp)
    X2 = range(10)
    Y2 = [(a2*s+b2) for s in X2]
    print('Stochastic Gradient Descent: y=%fX+%f'%(a2,b2))
 
    plt.xlabel("degree")
    plt.ylabel("Erms")
    plt.scatter(xTrain,yTrain, color = 'red',label = 'data')
    plt.plot(X1, Y1, color = 'blue',label = 'BGD')
    plt.plot(X2, Y2, color = 'green',label = 'SGD')
    plt.legend()
    plt.show()

###############Regression from order 0-9################# 
    M = np.arange(11)
    tr = np.zeros(len(M))
    te = np.zeros(len(M))
    for i in M:
        [w,tr[i]] = train_regre(i,xTrain,yTrain)
        te[i] = test_regre(i,xTest,yTest,w)
        print(w)
    print('\n')
    plt.xlabel("degree")
    plt.ylabel("Erms")
    plt.plot(M, tr, color = 'blue',marker='o',label = 'Training')
    plt.plot(M, te, color = 'red',marker='o',label = 'Test')
    plt.legend()
    plt.show()

###############Regularized regression################# 
    M = 9
    lamda = [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1]
    regtr = np.zeros(len(lamda))
    regte = np.zeros(len(lamda))
    for i in range(len(lamda)):
        [regw,regtr[i]] = train_reg(M,xTrain,yTrain,lamda[i]) 
        regte[i] = test_reg(M,xTest,yTest,regw) 
        print(regw)
    plt.xlabel("lamda")
    plt.ylabel("Erms")
    plt.plot(lamda, regtr, color = 'blue',label = 'Training')
    plt.plot(lamda, regte, color = 'red',label = 'Test')
    plt.legend()
    plt.show()
