import numpy as np

class NaiveBayes:
	# Naive Bayes implementation.
	# Assumes the feature are binary.
	# Also assumes the labels go from 0,1,...C-1

	def __init__(self, num_classes, beta=0):
		self.num_classes = num_classes
		self.beta = beta

	def fit(self, X, y):
		N, D = X.shape

		# Compute the number of class labels
		C = self.num_classes

		# Compute the probability of each class i.e p(y==c)
		# counts = np.bincount(y)

		counts = np.zeros(C)
		for c in range(1, C):
			counts[c] = np.count_nonzero(y == c)
		counts[0] = N - np.count_nonzero(y)
		p_y = counts / N

		# Compute the conditional probabilities i.e.
		# p(x(i,j)=1 | y(i)==c) as p_xy
		# p(x(i,j)=0 | y(i)==c) as p_xy
		p_xy = np.zeros([D, C])
		for c in range (C):
			for d in range (D):
				count = 0
				for n in range (N):
					if X[n, d] == 1 and y[n] == c:
						count += 1
				p_xy[d, c] = count / counts[c]		

		self.p_y = p_y
		self.p_xy = p_xy



	def predict(self, X):

		N, D = X.shape
		C = self.num_classes
		p_xy = self.p_xy
		p_y = self.p_y

		y_pred = np.zeros(N)
		for n in range(N):

			probs = p_y.copy() # initialize with the p(y) terms
			for d in range(D):
				if X[n, d] != 0:
					probs *= p_xy[d, :]
				else:
					probs *= (1-p_xy[d, :])

			y_pred[n] = np.argmax(probs)

		return y_pred
