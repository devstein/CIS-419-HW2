"""
======================
Script to Explore SVMs
======================

Simple script to explore SVM training with varying C

Example adapted from scikit_learn documentation by Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from svmKernels import myGaussianKernel
from svmKernels import _gaussSigma
from sklearn.metrics import accuracy_score


# load the data
filename = 'data/svmTuningData.dat'
allData = np.loadtxt(filename, delimiter=',')

X = allData[:,:-1]
Y = allData[:,-1]

C = 75

# train the SVM
print "Training the SVM"
equivalentGamma = 1.0 / (2 * 20	 ** 2)
model = svm.SVC(C = C, kernel='rbf', gamma=equivalentGamma)
model.fit(X, Y)

print accuracy_score(Y, model.predict(X))
h = .02  # step size in the mes

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

myPredictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
myPredictions = myPredictions.reshape(xx.shape)	


# Put the result into a color plot
plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
plt.title('SVM decision surface with C = '+str(C))
plt.axis('tight')
plt.show()

