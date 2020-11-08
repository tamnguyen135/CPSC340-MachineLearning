import numpy as np
from numpy.linalg import solve
from numpy import linalg as LA
import findMin
from scipy.optimize import approx_fprime
import utils
from numpy.random import permutation
from numpy.linalg import norm

class logReg:
    # Logistic Regression
    def __init__(self, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)


class logRegL2:
    # Logistic Regression
    def __init__(self, verbose=2, maxEvals=100, l = 0):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.l = l

    def funObj(self, w, X, y):
        l = self.l
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (l/2) * np.sum(np.square(w))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + l * w

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMin(self.funObj, self.w,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)


class logRegL1:
    # Logistic Regression
    def __init__(self, verbose=2, maxEvals=100, l = 0):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
        self.l = l

    def funObj(self, w, X, y):
        l = self.l
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = findMin.findMinL1( self.funObj, self.w, self.l,
                                      self.maxEvals, X, y, verbose=self.verbose)
    def predict(self, X):
        return np.sign(X@self.w)



class logRegL0(logReg):
    # L0 Regularized Logistic Regression
    def __init__(self, L0_lambda=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0_lambda = L0_lambda
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape
        minimize = lambda ind: findMin.findMin(self.funObj,
                                                  np.zeros(len(ind)),
                                                  self.maxEvals,
                                                  X[:, ind], y, verbose=0)
        selected = set()
        selected.add(0)
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        while minLoss != oldLoss:
            oldLoss = minLoss
            print("Epoch %d " % len(selected))
            print("Selected feature: %d" % (bestFeature))
            print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue

                selected_new = selected | {i} # tentatively add feature "i" to the seected set

                loss = minimize(np.array(list(selected_new)))[1]
                if loss < minLoss:
                    minLoss = loss
                    bestFeature = i
                # TODO for Q2.3: Fit the model with 'i' added to the features,
                # then compute the loss and update the minLoss/bestFeature


            selected.add(bestFeature)

        self.w = np.zeros(d)
        self.w[list(selected)], _ = minimize(list(selected))


class leastSquaresClassifier:
    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((self.n_classes,d))

        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # solve the normal equations
            # with a bit of regularization for numerical reasons
            self.W[i] = np.linalg.solve(X.T@X+0.0001*np.eye(d), X.T@ytmp)

    def predict(self, X):
        return np.argmax(X@self.W.T, axis=1)

class softmaxClassifier :
    def __init__ (self , verbose =0, maxEvals =100) :
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.bias = True
    
    def funObj (self , w, X, y):
        
        n, d = X.shape
        labels = np.unique (y)
        k = labels.size # num classes
        W = w.reshape ((k, d))
        ynorm = np.array ([np.argwhere(labels == c) [0][0] for c in y])
        
        # Calculate the function value
        f = np.sum(np.log(np.sum(np.exp(W.dot(x_i)))) for x_i in X)\
        - np.sum (np.multiply (X,[W[y_i] for y_i in ynorm]))
        
        # Calculate the gradient value
        # print([np.exp(W.dot(x_i)) for x_i in X])
        D = np.sum ([np.exp(W.dot(x_i)) for x_i in X], axis =1)
        # print(D)
        p = np.array ([np.divide (np.exp(X.dot(wc)),D) for wc in W])
        I = np.array ([ynorm == c for c in range (k)], dtype = float )
        pI = np.add (p, -I)
        g = pI@X
        
        return f, g.flatten()
    
    def fit(self , X, y):
        n, d = X.shape
        k = np.unique(y).size # num classes
        # Initial guess
        self.w = np.zeros (d * k)
        # utils.check_gradient (self , X, y)
        (self.w, f) = findMin.findMin(self.funObj , self.w, self.maxEvals , X, y, verbose = self.verbose)
        self.W = self.w.reshape(k, d)
    
    def predict (self , X):
        return np.argmax (X@self.W.T, axis =1)


class multiclassSVM():
    def __init__(self, epoch = 1, sigma= 0.5, lammy=1.0, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.sigma = sigma
        self.epoch = epoch

    def fit(self, X, y):
        n, d = X.shape
        self.n_classes = np.unique(y).size
        self.W =  0 * np.ones ((self.n_classes , d))
        for i in range(self.n_classes):
            print("class ", i)
            ybin = y.copy().astype(float)
            ybin[y==i] = 1
            ybin[y!=i] =  -1
            model = SVM(epoch = self.epoch, sigma = self.sigma, lammy = self.lammy, verbose=self.verbose , maxEvals=self.maxEvals)
            model.fit(X, ybin)
            self.W[i] = model.u

    def predict(self , X):
        return np.argmax(X@self.W.T, axis =1)


class SVM():
    def __init__(self, epoch = 1, sigma= 0.5, lammy=1.0, verbose=0, maxEvals=100):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.sigma = sigma
        self.epoch = epoch

    def fit(self, X, y):
        n, d = X.shape
        self.X = X
        # self.kernel_to_file()
        # utils.check_gradient(self, K, y, n, verbose=self.verbose)
        gamma0 = 0.0001
        t = 1
        self.u = np.ones(d)
        for m in range(self.epoch):
            # permute X randomly
            perm = permutation(len(X))
            for ind in range(n):

                i = perm[ind]

                gamma = gamma0

                if y[i] * (X[i] @ self.u) >= 1: 
                    self.u = (1 - gamma * self.lammy) * self.u    
                    # print("1st condition", self.u)
                else:    
                    self.u = (1 - gamma * self.lammy) * self.u + gamma * y[i] * X[i]
                    # print("2nd condition", self.u)

                gamma = gamma0 / (1 + t)
                t = t + 1   

        f = np.sum(np.maximum(np.zeros(n), np.add(np.ones(n), (- y * (X@self.u))))) + (1/2) * norm(self.u)
        print("loss : ", f)
        error = np.mean(np.sign(X@self.u) != y)
        print ("error: ", error)     

    def predict(self, Xtest):
        return np.sign(Xtest@self.u)
