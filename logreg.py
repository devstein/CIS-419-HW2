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

        for i in range(1,n):
            xi = X[i]
            yi = y[i]
            htheta = self.sigmoid(np.multiply(theta.T,xi))

            cost += yi * np.log(htheta) + (1 - yi) * np.log(1 - htheta)
            cost += (regLambda / 2) * reg
        cost = -cost
       
        #make sure cost isnt' a 1x1 matrix
        return cost.item(0)


        
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n,d = X.shape
        gradient = np.zeros((d))

        for j in range(d):
            for i in range(1,n):
                xi = X[i, :]
                yi = y[i]
                htheta = self.sigmoid(np.multiply(theta.T,xi))

                if (j != 0):
                    gradient[j] += (theta.item(0) - yi.item(0)) * X[i,j].item(0) + regLambda * theta[j].item(0)
                else:
                    print htheta, yi, (htheta - yi)
                    gradient[j] += htheta.item(0) - yi.item(0)

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

        self.theta = np.random.randn(d + 1)

        #stop iterations when ||theta new - theta old||2 <= epsioln
        #or we pass maxNumIters
        numIters = 0
        converged = False


        while(not converged and (numIters < self.maxNumIters)):

            thetaOld = np.copy(self.theta)

            self.theta = thetaOld - self.alpha * self.computeGradient(thetaOld, Xp, y, self.regLambda)

            converged = self.hasConverged(self.theta - thetaOld)

            numIters += 1



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
