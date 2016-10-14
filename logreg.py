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

        reg = 0

        for i in range(len(theta)):
            reg += np.linalg.norm(theta[i]) ** 2


        cost = 0

        X = np.asarray(X)  # same
        X = np.reshape(X, [n,d])  # reshape theta into a row vector, in case it comes is as a column vector
        y = np.asarray(y)
        y = np.reshape(y, [n,1])
        theta = np.asarray(theta)  # convert from np.matrix to np.array 
        theta = np.reshape(theta, [d,1])  # reshape theta into a col vector, in case it comes is as a column vector

        for i in range(1,n):
            xi = X[i]
            yi = y[i]
            htheta = self.sigmoid(np.dot(theta.T,xi))

            cost += yi * np.log(htheta) + (1 - yi) * np.log(1 - htheta)
        cost += (regLambda / 2) * reg
        cost= -cost

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
        gradient = np.reshape(gradient, [d,1])

        X = np.reshape(X, [n,d])  # reshape theta into a row vector, in case it comes is as a column vector
        X = np.reshape(X, [n,d])  # reshape theta into a row vector, in case it comes is as a column vector
        y = np.asarray(y)
        y = np.reshape(y, [n,1])
        theta = np.asarray(theta)  # convert from np.matrix to np.array 
        theta = np.reshape(theta, [d,1])  # reshape theta into a col vector, in case it comes is as a column vector

        # beacuse the auto grader hates my for loop
        # for j in range(0,d):
        #     for i in range(1,n):
        #         xi = X[i]
        #         yi = y[i]
        #         # d,1 x n, = 
        #         htheta = self.sigmoid(np.dot(xi, theta))

        #         if (j != 0):
        #             gradient[j] += (htheta - yi) * X[i,j] + regLambda * theta[j]

        #         else:
        #             gradient[j] += htheta - yi


        htheta = self.sigmoid(np.dot(X, theta))

        gradient = 1/n  * (np.dot(X.T, htheta - y) + regLambda * theta)

        gradient[0] = 0
        for i in range(1,n):
            h0 = self.sigmoid(np.dot(X[i], theta))
            gradient[0] += h0 - y[i]

        print gradient

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
        thetaNew = np.reshape(thetaNew, [d+1, 1]) 

        for numIters in range(self.maxNumIters):

            thetaOld = np.copy(thetaNew)

            thetaNew = thetaOld - (self.alpha * self.computeGradient(thetaOld, Xp, y, self.regLambda))

            if(self.hasConverged(thetaNew - thetaOld)):
                break

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


