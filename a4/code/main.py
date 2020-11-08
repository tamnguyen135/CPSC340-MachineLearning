import pickle
import gzip
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import findMin

import utils
import linear_model


from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from scipy.optimize import fmin

def f(l):
	def g(w):
		return 1/2 * w**2 - 2 * w + 5/2 + l * abs(w)**(1/2)
	return g


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-q','--question', required = True)
	io_args = parser.parse_args()
	question = io_args.question


	if question == "2":
		data = utils.load_dataset("logisticData")
		XBin, yBin = data['X'], data['y']
		XBinValid, yBinValid = data['Xvalid'], data['yvalid']

		model = linear_model.logReg(maxEvals=400)
		model.fit(XBin,yBin)

		print("\nlogReg Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
		print("logReg Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.w != 0).sum())

	elif question == "2.1":
		data = utils.load_dataset("logisticData")
		XBin, yBin = data['X'], data['y']
		XBinValid, yBinValid = data['Xvalid'], data['yvalid']

		model = linear_model.logRegL2(maxEvals=400, l=1.0)
		model.fit(XBin,yBin)

		print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
		print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.w != 0).sum())

	elif question == "2.2":
		data = utils.load_dataset("logisticData")
		XBin, yBin = data['X'], data['y']
		XBinValid, yBinValid = data['Xvalid'], data['yvalid']

		model = linear_model.logRegL1(l=1.0, maxEvals=400)
		model.fit(XBin,yBin)

		print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
		print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.w != 0).sum())

	elif question == "2.3":
		data = utils.load_dataset("logisticData")
		XBin, yBin = data['X'], data['y']
		XBinValid, yBinValid = data['Xvalid'], data['yvalid']

		model = linear_model.logRegL0(L0_lambda=1.0, maxEvals=400)
		model.fit(XBin,yBin)

		print("\nTraining error %.3f" % utils.classification_error(model.predict(XBin),yBin))
		print("Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.w != 0).sum())

	elif question == "2.5":
		data = utils.load_dataset("logisticData")
		XBin, yBin = data['X'], data['y']
		XBinValid, yBinValid = data['Xvalid'], data['yvalid']

		


		model = linear_model.logRegL2(maxEvals=400, l=1.0)
		model.fit(XBin,yBin)

		print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
		print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.w != 0).sum())

		model = LogisticRegression(penalty = 'l2', fit_intercept = False).fit(XBin, yBin)	

		print("\nlogRegL2 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
		print("logRegL2 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.coef_ != 0).sum())

		



		model = linear_model.logRegL1(l=1.0, maxEvals=400)
		model.fit(XBin,yBin)

		print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin),yBin))
		print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.w != 0).sum())

		model = LogisticRegression(penalty = 'l1', fit_intercept = False).fit(XBin, yBin)	

		print("\nlogRegL1 Training error %.3f" % utils.classification_error(model.predict(XBin), yBin))
		print("logRegL1 Validation error %.3f" % utils.classification_error(model.predict(XBinValid), yBinValid))
		print("# nonZeros: %d" % (model.coef_ != 0).sum())

	elif question == "2.6":
		w = np.linspace(-10, 10, 1000)
		y = 1/2 * w**2 - 2 * w + 5/2 + abs(w)**(1/2)
		print(fmin(f(1), 0))
		plt.plot(w, y)
		plt.xlabel("w")
		plt.ylabel("f(w)")
		fname = os.path.join("..", "figs", "q2_6_lambda_1.pdf")
		plt.savefig(fname)

	elif question == "2.6.1":
		w = np.linspace(-10, 10, 1000)
		y = 1/2 * w**2 - 2 * w + 5/2 + 10 * abs(w)**(1/2)
		print(fmin(f(10), 0))
		plt.plot(w, y)
		plt.xlabel("w")
		plt.ylabel("f(w)")
		fname = os.path.join("..", "figs", "q2_6_lambda_10.pdf")
		plt.savefig(fname)

	elif question == "3":
		data = utils.load_dataset("multiData")
		XMulti, yMulti = data['X'], data['y']
		XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

		print(XMulti.shape, XMultiValid.shape)

		model = linear_model.multiclassSVM(epoch = 1000, lammy = 0.01, maxEvals = 1000)
		model.fit(XMulti, yMulti)

		print("multiclassSVM Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
		print("multiclassSVM Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

		print(np.unique(model.predict(XMulti)))


	elif question == "3.2":
		data = utils.load_dataset("multiData")
		XMulti, yMulti = data['X'], data['y']
		XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

		model = linear_model.logLinearClassifier(maxEvals=500, verbose=0)
		model.fit(XMulti, yMulti)

		print("logLinearClassifier Training error %.3f" % utils.classification_error(model.predict(XMulti), yMulti))
		print("logLinearClassifier Validation error %.3f" % utils.classification_error(model.predict(XMultiValid), yMultiValid))

	elif question == "3.4":
		with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f: 
			train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
		X, y = train_set
		Xtest, ytest = test_set

		binarizer = LabelBinarizer()
		Y = binarizer.fit_transform(y)

		model = linear_model.softmaxClassifier(maxEvals=500)
		model.fit(X, y)

		print("Testing error %.3f" % utils.classification_error(model.predict(Xtest), ytest))
	
	elif question == "3.5":
		data = utils.load_dataset("multiData")
		XMulti, yMulti = data['X'], data['y']
		XMultiValid, yMultiValid = data['Xvalid'], data['yvalid']

		