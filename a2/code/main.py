# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN

def load_dataset(filename):
	with open(os.path.join('..','data',filename), 'rb') as f:
		return pickle.load(f)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-q','--question', required=True)

	io_args = parser.parse_args()
	question = io_args.question


	if question == "1":
		with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
			dataset = pickle.load(f)

		X, y = dataset["X"], dataset["y"]
		X_test, y_test = dataset["Xtest"], dataset["ytest"]        
		model = DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=1)
		model.fit(X, y)

		y_pred = model.predict(X)
		tr_error = np.mean(y_pred != y)

		y_pred = model.predict(X_test)
		te_error = np.mean(y_pred != y_test)
		print("Training error: %.3f" % tr_error)
		print("Testing error: %.3f" % te_error)

	elif question == "1.1":
		
		with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
			dataset = pickle.load(f)

		X, y = dataset["X"], dataset["y"]
		X_test, y_test = dataset["Xtest"], dataset["ytest"]

		depths = np.arange(1,16)

		tr_errors = np.zeros(depths.size)
		te_errors = np.zeros(depths.size)

		for i, max_depth in enumerate(depths):  
			model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
			model.fit(X, y)

			y_pred = model.predict(X)
			tr_errors[i] = np.mean(y_pred != y)

			y_pred = model.predict(X_test)
			te_errors[i] = np.mean(y_pred != y_test)

			print("depth" + str(max_depth) + "training error: %.3f" % tr_errors[i])
			print("depth" + str(max_depth) + "testing error: %.3f" % te_errors[i])

		
		plt.plot(depths, tr_errors, label = "training error")
		plt.plot(depths, te_errors, label = "test error")	

		plt.xlabel("Depth of tree")
		plt.ylabel("Classification error")
		plt.legend()
		fname = os.path.join("..", "figs", "q1_training_vs_test_errors.pdf")
		plt.savefig(fname)	

	


	elif question == '1.2':
		
		with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
			dataset = pickle.load(f)

		X, y = dataset["X"], dataset["y"]
		n, d = X.shape
		
		print("Use first half as training data and second half as validation set")

		X_train = X[:round(n/2)]
		y_train = y[:round(n/2)]
		X_valid = X[round(n/2):]
		y_valid = y[round(n/2):]

		depths = np.arange(1,16)

		for i, max_depth in enumerate(depths):  
			model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
			model.fit(X_train, y_train)

			y_pred = model.predict(X_valid)
			error = np.mean(y_pred != y_valid)

			print("Depth " + str(max_depth) + " training error: %.3f" % error)


		print("Use second half as training data and first half as validation set")	

		X_valid = X[:round(n/2)]
		y_valid = y[:round(n/2)]
		X_train = X[round(n/2):]
		y_train = y[round(n/2):]

		depths = np.arange(1,16)

		for i, max_depth in enumerate(depths):  
			model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
			model.fit(X_train, y_train)

			y_pred = model.predict(X_valid)
			error = np.mean(y_pred != y_valid)

			print("Depth " + str(max_depth) + " training error: %.3f" % error)

		


	elif question == '2.2':
		dataset = load_dataset("newsgroups.pkl")

		X = dataset["X"]
		y = dataset["y"]
		X_valid = dataset["Xvalidate"]
		y_valid = dataset["yvalidate"]
		groupnames = dataset["groupnames"]
		wordlist = dataset["wordlist"]
		
		n, d = X.shape

		print("column 50 word: " + wordlist[50])

		print("words in X[500]: ")

		for i in range(d):
			if X[500][i] == 1:
				print(wordlist[i])

		print("newsgroup of X[500]: " + groupnames[y[500]])





	elif question == '2.3':
		dataset = load_dataset("newsgroups.pkl")

		X = dataset["X"]
		y = dataset["y"]
		X_valid = dataset["Xvalidate"]
		y_valid = dataset["yvalidate"]

		print("d = %d" % X.shape[1])
		print("n = %d" % X.shape[0])
		print("t = %d" % X_valid.shape[0])
		print("Num classes = %d" % len(np.unique(y)))

		model = NaiveBayes(num_classes=4)
		model.fit(X, y)
		y_pred = model.predict(X_valid)
		v_error = np.mean(y_pred != y_valid)
		print("Naive Bayes (ours) validation error: %.3f" % v_error)

		clf = BernoulliNB()
		clf.fit(X,y)
		y_predd = clf.predict(X_valid)
		v_errorr = np.mean(y_predd != y_valid)
		print("Naive Bayes (ours) validation error: %.3f" % v_errorr)

	

	elif question == '3':
		with open(os.path.join('..','data','citiesSmall.pkl'), 'rb') as f:
			dataset = pickle.load(f)

		X = dataset['X']
		y = dataset['y']
		Xtest = dataset['Xtest']
		ytest = dataset['ytest']

		k = 1
		model = KNN(k = k)
		model.fit(X, y)
		y_pred = model.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours (ours) test error: %.3f" % error)
		knc = KNeighborsClassifier(n_neighbors = k)
		knc.fit(X, y)
		y_pred = knc.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours test error: %.3f" % error)
		utils.plotClassifier(model, Xtest, ytest)
		fname = os.path.join("..", "figs", "q3_our_KNN.pdf")
		plt.savefig(fname)	
		utils.plotClassifier(knc, Xtest, ytest)
		fname = os.path.join("..", "figs", "q3_sklearn_KNN.pdf")
		plt.savefig(fname)	
		
		

		k = 3
		model = KNN(k = k)
		model.fit(X, y)
		y_pred = model.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours (ours) test error: %.3f" % error)
		knc = KNeighborsClassifier(n_neighbors = k)
		knc.fit(X, y)
		y_pred = knc.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours test error: %.3f" % error)
		

		k = 10
		model = KNN(k = k)
		model.fit(X, y)
		y_pred = model.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours (ours) test error: %.3f" % error)
		knc = KNeighborsClassifier(n_neighbors = k)
		knc.fit(X, y)
		y_pred = knc.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours test error: %.3f" % error)
		
		

		k = 1
		model = KNN(k = k)
		model.fit(X, y)
		y_pred = model.predict(X)
		error = np.mean(y_pred != y)
		print (str(k) + " nearest Neighbours (ours) training error: %.3f" % error)
		knc = KNeighborsClassifier(n_neighbors = k)
		knc.fit(X, y)
		y_pred = knc.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours training error: %.3f" % error)
		
		

		k = 3
		model = KNN(k = k)
		model.fit(X, y)
		y_pred = model.predict(X)
		error = np.mean(y_pred != y)
		print (str(k) + " nearest Neighbours (ours) training error: %.3f" % error)
		knc = KNeighborsClassifier(n_neighbors = k)
		knc.fit(X, y)
		y_pred = knc.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours training error: %.3f" % error)
		
		

		k = 10
		model = KNN(k = k)
		model.fit(X, y)
		y_pred = model.predict(X)
		error = np.mean(y_pred != y)
		print (str(k) + " nearest Neighbours (ours) training error: %.3f" % error)
		knc = KNeighborsClassifier(n_neighbors = k)
		knc.fit(X, y)
		y_pred = knc.predict(Xtest)
		error = np.mean(y_pred != ytest)
		print (str(k) + " nearest Neighbours training error: %.3f" % error)
		
		
		
	elif question == '4':
		dataset = load_dataset('vowel.pkl')
		X = dataset['X']
		y = dataset['y']
		X_test = dataset['Xtest']
		y_test = dataset['ytest']
		print("\nn = %d, d = %d\n" % X.shape)

		def evaluate_model(model):
			model.fit(X,y)

			y_pred = model.predict(X)
			tr_error = np.mean(y_pred != y)

			y_pred = model.predict(X_test)
			te_error = np.mean(y_pred != y_test)
			print("    Training error: %.3f" % tr_error)
			print("    Testing error: %.3f" % te_error)

		def time_model(model):
			t = time.time()
			model.fit(X,y)

			y_pred = model.predict(X)
			tr_error = np.mean(y_pred != y)

			y_pred = model.predict(X_test)
			te_error = np.mean(y_pred != y_test)

			print("This took %f seconds" % (time.time()-t))

			print("    Training error: %.3f" % tr_error)
			print("    Testing error: %.3f" % te_error)

		print("Decision tree info gain")
		evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
		print("Random forest")
		evaluate_model(RandomForest(max_depth=np.inf, num_trees=50))
		print("Random tree")
		evaluate_model(RandomTree(max_depth=np.inf))

		time_model(RandomForestClassifier(max_depth=100, n_estimators = 50))
		time_model(RandomForest(max_depth=100, num_trees=50))
		


	elif question == '5':
		X = load_dataset('clusterData.pkl')['X']

		model = Kmeans(k=4)
		model.fit(X)
		y = model.predict(X)
		plt.scatter(X[:,0], X[:,1], c=y, cmap="jet")

		fname = os.path.join("..", "figs", "kmeans_basic.png")
		plt.savefig(fname)
		print("\nFigure saved as '%s'" % fname)

	elif question == '5.1':
		X = load_dataset('clusterData.pkl')['X']



	elif question == '5.2':
		X = load_dataset('clusterData.pkl')['X']



	elif question == '5.3':
		X = load_dataset('clusterData2.pkl')['X']

		model = DBSCAN(eps=1, min_samples=3)
		y = model.fit_predict(X)

		print("Labels (-1 is unassigned):", np.unique(model.labels_))
		
		plt.scatter(X[:,0], X[:,1], c=y, cmap="jet", s=5)
		fname = os.path.join("..", "figs", "density.png")
		plt.savefig(fname)
		print("\nFigure saved as '%s'" % fname)
		
		plt.xlim(-25,25)
		plt.ylim(-15,30)
		fname = os.path.join("..", "figs", "density2.png")
		plt.savefig(fname)
		print("Figure saved as '%s'" % fname)
		
	else:
		print("Unknown question: %s" % question)
