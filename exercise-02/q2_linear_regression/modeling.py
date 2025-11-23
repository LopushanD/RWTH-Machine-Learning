import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional

def normalize(x: NDArray,
              m: Optional[NDArray]=None,
              std: Optional[NDArray]=None) -> Tuple[NDArray, NDArray, NDArray]:
    """Normalize `x` such that it has mean `m` and standard deviation
    `std`. If these equal None, `x` is normalized to have zero mean and
    unit standard deviation.

    Returns:
        A tuple consisting of the normalized version of `x`, the mean 
        that was used for normalizing, and the standard deviation that
        was used for normalizing.
    """
    if(m is None):
        m = np.mean(x)
    if(std is None):
        std = np.std(x)
    return (x - m)/std, m, std


def fit_linear_model(x: NDArray, y: NDArray) -> Tuple[NDArray, NDArray]:
    """Fit a linear regression model.

    Args:
        x: a numpy array of shape (`N`, `D_x`), or of shape (`N`,), 
            corresponding to the independent variable. In both 
            cases, `N` is the number of samples. In the first case, `D_x`
            is the feature dimension, and in the second, `D` is assumed
            to be 1.
        y: a numpy array of shape (`N`,), corresponding to the dependent 
            variable. `N` is again the number of samples.
    Returns:
        A tuple of two numpy arrays, one containing `D_x` weights and 
        one containing the bias (i.e., an array with a single element).
    """
    if(len(x.shape) > 1):
        num_samples, x_feature_dim = x.shape
    else:
        num_samples, x_feature_dim = x.shape[0], 1
        #insert a dummy dim to allow the remaining code to work in the 
        #same way for both cases
        x = x[:, None]

    if(y.shape[0] != num_samples):
        raise ValueError(
            "x, y must have the same number of rows (= number of samples)")
    #below you should make sure to prepend or append a column of 1s
    #(bias trick) and then implement the lecture's formula

    #####Insert your code here for subtask 2a#####
    x_0 = np.ones((num_samples,1))
    x_hat  = np.hstack((x,x_0))
    
    rng = np.random.default_rng()
    w = rng.random(x_hat.shape)
    
    pseudo_inverse = np.linalg.pinv(x_hat)
    w = np.dot(pseudo_inverse,y)
        
    assert w.shape == (x_hat.shape[1],)
    return w[:-1], w[-1]

def eval_linear(x: NDArray, w: NDArray, b: NDArray) -> NDArray:
    """Apply a linear model with weights `w` and bias `b` on input data
    `x` of shape (N, D) or shape (N,), `N` the number of samples and
    `D` the feature dimensionality (assumed 1 in the second case).
    """
    if(len(x.shape) > 1):
        num_samples, x_feature_dim = x.shape
    else:
        num_samples, x_feature_dim = x.shape[0], 1
        x = x[:, None]
    
    return w.T @ x.T + b

def compute_polynomial_basis_funcs(x: NDArray, 
                                   highest_power: int) -> NDArray:
    """Compute the values of the polynomial basis functions 
    `v, v^2, ..., v^M`, where `M=highest_power` on an input of shape
    `(N,)`, assumed to be `N` single-dimensional samples.

    Returns:
        An array of shape `(N, highest_power)` whose i-th row contains
        the values of the polynomial basis functions evaluated on the
        i-th sample.
    """
    
    num_samples = x.shape[0]
    if(len(x.shape) > 1):
        raise ValueError("For simplicity, compute_polynomial_basis_funcs "
                         "is only meant to work on 1-D input")
    #NOTE: don't prepend or append a column of ones! The fitting
    #functions take care of this

    #####Insert your code here for subtask 2b#####

    assert phi_of_x.shape == (num_samples, highest_power)
    return phi_of_x

def eval_polynomial(x: NDArray, highest_power: int, 
                    w: NDArray, b: NDArray) -> NDArray:
    """Apply a linear model with weights `w` and bias `b` on input data
    `x` of shape `(N,)`, i.e. `N` single-dimensional samples, after
    mapping `x` to a `highest_power`-dimensional space via polynomial
    basis functions of degree `highest_power`.

    `w` should be a `highest_power`-dimensional vector, since it's used
    in the space resulting from the polynomial basis functions.
    """
    
    phi_of_x = compute_polynomial_basis_funcs(x, highest_power)
    #note how we can now use the same function as for linear eval
    return eval_linear(phi_of_x, w, b)
    

def fit_linear_model_with_ridge_regression(
        x: NDArray, 
        y: NDArray, ridge_lambda: float) -> Tuple[NDArray, NDArray]:
    """Fit a linear regression model using l-2 regularization (ridge
    regression).

    Args:
        x: a numpy array of shape (`N`, `D_x`), or of shape (`N`,), 
            corresponding to the independent variable. In both 
            cases, `N` is the number of samples. In the first case, `D_x`
            is the feature dimension, and in the second, `D` is assumed
            to be 1. NOTE: The data is assumed to have been previously 
            normalized to have zero mean.
        y: a numpy array of shape (`N`,), corresponding to the dependent 
            variable. `N` is again the number of samples.
        ridge_lambda: the strength of regularization.
    Returns:
        A tuple of two numpy arrays, one containing `D_x` weights and 
        one containing the bias (i.e., an array with a single element).
    """    
    if(len(x.shape) > 1):
        num_samples, x_feature_dim = x.shape
    else:
        num_samples, x_feature_dim = x.shape[0], 1
        x = x[:, None]

    if(y.shape[0] != num_samples):
        raise ValueError(
            "x, y must have the same number of rows (= number of samples)")
    #below you should make sure to prepend or append a column of 1s
    #(bias trick) and then implement the lecture's formula    

    #####Insert your code here for subtask 2c#####
    assert w.shape == (x.shape[1],)
    return w, bias

