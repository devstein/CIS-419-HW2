"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 3
_gaussSigma = 1000

def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''

    return (np.inner(X1,X2) + 1) ** _polyDegree



def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1 = len(X1)
    n2 = len(X2)

    G = np.zeros((n1,n2))

    for i in range(n1):
        for j in range(n2):
            G[i, j] += np.linalg.norm(X1[i] - X2[j]) ** 2

    denom = 2 * (_gaussSigma ** 2)
    return np.exp(-G / denom)



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    return #TODO (CIS 519 ONLY)

