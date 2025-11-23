import numpy as np

def leastSquares(data, label):
    # Sum of squared error shoud be minimized
    #
    # INPUT:
    # data        : Training inputs  (num_samples x dim)
    # label       : Training targets (num_samples x 1)
    #
    # OUTPUT:
    # weights     : weights   (dim x 1)
    # bias        : bias term (scalar)
    
    #####Insert your code here for subtask 1a#####
    # Extend each datapoint x as [1, x]
    # (Trick to avoid modeling the bias term explicitly)
    rng = np.random.default_rng()
    w = rng.random((3)) # DO NOT FORGET BIAS AS ADDITIONAL PARAMETER
    
    # add bias' value to data (always 1)
    biases = np.ones((data.shape[0],1))
    x = np.hstack((data,biases))
    
    # closed form solution
    x_preudo_inverse = np.linalg.pinv(x) 
    w = np.dot(x_preudo_inverse,label)
    
    #gradient descent
    # for i in range (600):
    #     # pass forward
    #     prediction = np.dot(x,w.T)
    #     loss = 0.5*sum((prediction-label)**2)
    #     print(loss)
    #     #backward pass
    #     update = -0.001*np.dot(x.T,(prediction-label))
    #     w+=update 
           
    return w[0:2],w[2]
