'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''
from __future__ import division
import numpy as np


class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n,d = X.shape

        reg = np.linalg.norm(theta) ** 2

        cost = 0

        X = np.asarray(X)  # same
        X = np.reshape(X, [n,d])  # reshape theta into a row vector, in case it comes is as a column vector
        y = np.asarray(y)
        y = np.reshape(y, [n,1])
        theta = np.asarray(theta)  # convert from np.matrix to np.array 
        theta = np.reshape(theta, [1,d])  # reshape theta into a row vector, in case it comes is as a column vector

        for i in range(1,n):
            xi = X[i]
            yi = y[i]
            htheta = self.sigmoid(np.dot(theta,xi))

            cost += yi * np.log(htheta) + (1 - yi) * np.log(1 - htheta)
            cost += (regLambda / 2) * reg
        cost = -cost
       
        #make sure cost isnt' a 1x1 matrix
        return cost.item(0)

        
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d    numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n,d = X.shape
        gradient = np.zeros((d))

        X = np.reshape(X, [n,d])  # reshape theta into a row vector, in case it comes is as a column vector
        X = np.reshape(X, [n,d])  # reshape theta into a row vector, in case it comes is as a column vector
        y = np.asarray(y)
        y = np.reshape(y, [n,1])
        theta = np.asarray(theta)  # convert from np.matrix to np.array 
        theta = np.reshape(theta, [1,d])  # reshape theta into a row vector, in case it comes is as a column vector

        for j in range(d):
            for i in range(1,n):
                xi = X[i, :]
                yi = y[i]
                htheta = self.sigmoid(np.dot(theta,xi))

                if (j != 0):
                    gradient[j] += (htheta - yi) * X[i,j] + regLambda * theta[0,j]
                else:
                    gradient[j] += htheta - yi

        return gradient

    def sigmoid(self, Z):
        '''
        Computers the sigmoid function 1/(1+exp(-z))
        '''
        return 1 / (1 + np.exp(-Z ))  



    def hasConverged(self, theta):
        return np.linalg.norm(theta) <= self.epsilon

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n,d = X.shape 

        Xp = np.c_[np.ones((n,1)),X]

        thetaNew = np.random.randn(d + 1)

        #stop iterations when ||theta new - theta old||2 <= epsioln
        #or we pass maxNumIters
        numIters = 0
        converged = False


        while(not converged and (numIters < self.maxNumIters)):

            thetaOld = np.copy(thetaNew)

            thetaNew = thetaOld - self.alpha * self.computeGradient(thetaOld, Xp, y, self.regLambda)

            converged = self.hasConverged(thetaNew - thetaOld)

            numIters += 1

        print numIters
        self.theta = thetaNew

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n = len(X)
        
        # add 1s column
        Xp = np.c_[np.ones([n, 1]), X];

        # predict
        return np.array(np.around(self.sigmoid(Xp.dot(self.theta))))


