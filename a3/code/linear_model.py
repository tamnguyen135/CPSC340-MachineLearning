import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        ''' YOUR CODE HERE '''
        raise NotImplementedError()

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE '''
        # Calculate the function value
        f = 0.5*np.sum((X@w - y)**2)

        # Calculate the gradient value
        g = X.T@(X@w-y)

        return (f,g)

    def predict(self, X):
        return X@self.w


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        n, d = X.shape
        Z = np.append(X, np.ones((n, 1)), axis = 1)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        n, d = X.shape
        Z = np.append(X, np.ones((n, 1)), axis = 1)
        return Z@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.leastSquares.fit(Z, y)

    def predict(self, X):
        return self.leastSquares.predict(self.__polyBasis(X))

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        n, d = X.shape
        Z = np.ones([n, 1])
        p = self.p
        for j in np.arange(1, p + 1):
            Z = np.append(Z, X**j, axis = 1)
        return Z