import numpy as np

class Linear:
    def __init__(self,inputShape:tuple[int,int],learningRate,numberOfLayers):
        #TODO: recommended to generate biases as small positive numbers
        # self.W = np.random.random((numberOfLayers,inputShape[0],inputShape[1]+1))
        self.W = np.array([[1,2,3],[1,4,5]])
        self.inputCache = None
        self.dW = None
        self.lr = learningRate

    def _addBias(self,X):
        return np.hstack([np.ones((X.shape[0],1)),X]) 
    
    def fprop(self,X):
       X = self._addBias(X)
       print(f"X:\n{X}")
       print(f"W:\n{self.W}")
       print(f"W.T:\n{self.W.T}")
       self.inputCache = X
       return np.dot(X,self.W.T)

    def bprop(self,dE):
        # seems like dE is actually dE/dy 
        self.dW = np.multiply(self.inputCache,dE) # update weights
        print(f"dW:\n{self.dW}")
        #finish this return statement
        return np.multiply(None,dE.T) # change of weights in current layer w.r.t inputs to current layer

                

m = Linear((2,2),0.001)

arr = np.array([[2,3],[4,5]])
res = m.fprop(arr)
print(res)
m.bprop(np.array([[4,-1]]).T)