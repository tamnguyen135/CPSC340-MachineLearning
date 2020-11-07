"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
    	X = self.X
    	y = self.y
    	k = self.k
    	distances = utils.euclidean_dist_squared(X, Xtest)
    	N, T = distances.shape
    	y_pred = np.zeros(T)
    	for t in range (T):
    		y_near = y[distances[:, t].argsort()]
    		y_pred[t] = utils.mode(y_near[:k])
    	return y_pred

    		