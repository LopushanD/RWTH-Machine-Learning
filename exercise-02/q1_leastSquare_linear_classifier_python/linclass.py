import numpy as np
def linclass(weight, bias_value, data):
    # Linear Classifier
    #
    # INPUT:
    # weight      : weights                (dim x 1)
    # bias        : bias term              (scalar)
    # data        : Input to be classified (num_samples x dim)
    #
    # OUTPUT:
    # class_pred       : Predicted class (+-1) values  (num_samples x 1)

    #####Insert your code here for subtask 1b#####
    w = np.hstack((weight,bias_value))
    bias_value = np.ones((data.shape[0],1))
    x = np.hstack((data,bias_value))
    prediction:np.ndarray = np.dot(x,w.T)
    # Perform linear classification i.e. class prediction
    class_pred = np.where(prediction>=0,1,-1)
    return class_pred


