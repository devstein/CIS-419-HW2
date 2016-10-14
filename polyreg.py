'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

from __future__ import division
import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        self.degree = degree;
        self.regLambda = regLambda;

    def getMeans(self, features):
        n,d = features.shape
        means = np.zeros((d))

        for i in range(d):
            means[i] = np.mean(features[:,i])
        
        self.means = means

    def getStds(self, features):
        n,d = features.shape
        stds = np.zeros((d))

        for i in range(d):
           stds[i] = np.std(features[:,i])
        
        self.stds = stds

    def standardize(self, features):
        standardized = np.zeros(features.shape)

        n,d = features.shape

        for j in range(d):
            for i in range(n):
                if (self.stds[j] != 0.0): 
                    standardized[i,j] = (features[i,j] - self.means[j]) / self.stds[j]
                else:
                    standardized[i,j] =  features[i,j]

        return standardized



    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.
        
        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''

        features = np.zeros((len(X), degree));

        for i, x in enumerate(X):
            for j in range(degree):
                features[i,j] = x ** (j+1)

        return features
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        features = self.polyfeatures(X, self.degree)

        n,d = features.shape

        self.getMeans(features)
        self.getStds(features)

        standardized = self.standardize(features)

        Xp = np.c_[np.ones([n,1]), standardized]


        regMatrix = self.regLambda * np.eye(d + 1)
        regMatrix[0,0] = 0

        self.theta = np.linalg.pinv(Xp.T.dot(Xp) + regMatrix).dot(Xp.T).dot(y)
                
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        n = len(X)
        
        features = self.polyfeatures(X, self.degree)

        standardized = self.standardize(features)
        # add 1s column
        Xp = np.c_[np.ones([n, 1]), standardized];

        # predict
        return Xp.dot(self.theta)

#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def regressionError(h, y):

    n = len(h)
    sum = 0
    if (n == 0):
        return 0

    for i in range(n):
        sum = sum + (h[i] - y[i]) ** 2

    return sum/n


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape

    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    
    for i in range(1,n):
        model = PolynomialRegression(degree = degree, regLambda = regLambda)
        model.fit(Xtrain[0:(i+1)], Ytrain[0:(i+1)])

        errorTrain[i] = regressionError(model.predict(Xtrain[0:(i+1)]), Ytrain[0:(i+1)])
        errorTest[i] = regressionError(model.predict(Xtest[0:(i+1)]), Ytest[0:(i+1)])

    
    return (errorTrain, errorTest)

