import numpy as np

def applyKernel(kernelSize,sample):
    return 1/((2*np.pi*kernelSize**2)**0.5)*np.exp(-(sample**2)/(2*kernelSize**2))

def kde(samples, h):
    # compute density estimation from samples with KDE
    # Input
    #  samples    : DxN matrix of data points
    #  h          : (half) window size/radius of kernel
    # Output
    #  estDensity : estimated density in the range of [-5,5]

    #####Insert your code here for subtask 5a#####
    pos = np.arange(-5, 5.0, 0.1) # Returns a 100 dimensional vector
    N = len(samples)
    density = []
    for x in pos:
        sigma = []
        for x_n in samples:
            sigma.append(applyKernel(h,(x-x_n)))
        averaged = np.sum(sigma)/N
        density.append(averaged)
    
    #remake according to solution
    # norm = np.sqrt(2*np.pi)*h*N
    # res = np.sum(np.exp(-(pos[np.newaxis, :] - samples[:, np.newaxis]) ** 2 / (2 * h ** 2)), axis=0) / norm
    # estDensity = np.stack((pos,res),axis=1)
    #
    
    estDensity = np.stack((pos,density),axis=1)
    # Compute the number of samples created
    return estDensity