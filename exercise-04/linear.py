import numpy as np
import pandas as pd
import random
import seaborn as sns
class Linear:
    def __init__(self,inputLength:tuple[int,int],learningRate,numberOfLayers):
        #TODO: recommended to generate biases as small positive numbers
        self.W = np.random.random((numberOfLayers,inputLength+1))
        # self.W = np.array([[1,2,3],[1,4,5]])
        self.numberOfLayers = numberOfLayers
        self.inputCache = None
        self.dW = None
        self.lr = learningRate

    def _addBias(self,X):
        return np.hstack([np.ones(1),X]) 
    
    def fprop(self,X):
       X = self._addBias(X)
    #    print(f"X:\n{X}")
    #    print(f"W:\n{self.W}")
    #    print(f"W.T:\n{self.W.T}")
       self.inputCache = X
       return np.dot(X,self.W.T)

    def bprop(self,dE):
        # seems like dE is actually dE/dy 
        self.dW = np.multiply(self.inputCache,dE) # update weights
        print(f"dW:\n{self.dW}")
        #finish this return statement
        return np.multiply(self.W,dE.T) # change of weights in current layer w.r.t inputs to current layer

    def update(self,lr):
        self.W -= lr*self.dW
                
class Softmax:
    def __init__(self):
        self.cache = None
    
    def fprop(self,A):
        exp = np.exp(A)
        self.cache = exp / np.sum(exp)
        return self.cache
    
    def bprop(self,dE):
        y = self.cache
        dot = np.sum(dE*y)
        dZ = y * (dE - dot)
        return dZ
    
class CrossEntropy:
    def  __init__(self,targets):
        self.targets = targets
        self.cache = None
        
    def fprop(self,X):
        eps = 1e-12
        self.cache = np.clip(X,eps,1)
        print("shape:", self.cache.shape)
        print("targets shape:", self.targets.shape)
        return -np.sum(np.multiply(self.targets,np.log(self.cache)))
    
    def bprop(self):
        return -(self.targets/self.cache)
    
train_data = pd.read_csv("exercise-04\mnist\mnist-train-data.csv",sep=" ",header=None)
train_labels = pd.read_csv("exercise-04\mnist\mnist-train-labels.csv",header=None)
datasetLength = len(train_data)
epochs = 10
linear = Linear(28*28,0.01,10)
sm = Softmax()
ce = CrossEntropy(train_labels)
losses =[]
for _ in range(epochs):
    randIndex = random.randint(0,datasetLength-1)
    data = train_data.loc[randIndex,:].to_numpy()
    label = train_labels.loc[randIndex,:].to_numpy()
    # break
    # forward
    x = linear.fprop(data)
    x = sm.fprop(x)
    loss = ce.fprop(x)
    losses.append(loss)
    #backward
    dE = ce.bprop()
    dE = sm.bprop(dE)
    dE = linear.bprop(dE)
 
sns.lineplot(losses)