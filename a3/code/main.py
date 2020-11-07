
# basics
import argparse
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix as sparse_matrix

# sklearn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

# our code
import linear_model
import utils

url_amazon = "https://www.amazon.com/dp/%s"

def load_dataset(filename):
	with open(os.path.join('..','data',filename), 'rb') as f:
		return pickle.load(f)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-q','--question', required=True)
	io_args = parser.parse_args()
	question = io_args.question

	if question == "1":

		filename = "ratings_Patio_Lawn_and_Garden.csv"
		with open(os.path.join("..", "data", filename), "rb") as f:
			ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))

		print("Number of ratings:", len(ratings))
		print("The average rating:", np.mean(ratings["rating"]))

		n = len(set(ratings["user"]))
		d = len(set(ratings["item"]))
		print("Number of users:", n)
		print("Number of items:", d)
		print("Fraction nonzero:", len(ratings)/(n*d))

		X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
		print(type(X))
		print("Dimensions of X:", X.shape)


	elif question == "1.1":
		filename = "ratings_Patio_Lawn_and_Garden.csv"
		with open(os.path.join("..", "data", filename), "rb") as f:
			ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
		X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
		X_binary = X != 0
		
		n, d = X.shape

		# YOUR CODE HERE FOR Q1.1.1
		
		# y = X.sum(axis = 0)
		# print (item_inverse_mapper[y.argmax()] + " " + str(y.max()))
		

		# Q1.1.2

		# x = X_binary.sum(axis = 1)
		# print (user_inverse_mapper[x.argmax()] + " " + str(x.max()))		

		
		# YOUR CODE HERE FOR Q1.1.3
		# ratings_per_item = X.getnnz(axis = 0)
		# num_bins = 100
		# plt.hist(ratings_per_item, num_bins, facecolor='blue')
		# plt.yscale('log', nonposy='clip')
		# plt.xlabel("Ratings per item")
		# plt.ylabel("Number of items")
		# fname = os.path.join("..", "figs", "q1_ratings_per_item.pdf")
		# plt.savefig(fname)
		
		# ratings_per_user = X.getnnz(axis = 1)
		# num_bins = 100
		# plt.hist(ratings_per_user, num_bins, facecolor='blue')
		# plt.yscale('log', nonposy='clip')
		# plt.xlabel("Ratings per user")
		# plt.ylabel("Number of users")
		# fname = os.path.join("..", "figs", "q1_ratings_per_user.pdf")
		# plt.savefig(fname)
		
		num_bins = 4
		plt.hist(ratings["rating"], num_bins, facecolor='blue')
		plt.yscale('log', nonposy='clip')
		plt.xlabel("Rating")
		plt.ylabel("Number of ratings")
		fname = os.path.join("..", "figs", "q1_ratings.pdf")
		plt.savefig(fname)
		


	elif question == "1.2":
		filename = "ratings_Patio_Lawn_and_Garden.csv"
		with open(os.path.join("..", "data", filename), "rb") as f:
			ratings = pd.read_csv(f,names=("user","item","rating","timestamp"))
		X, user_mapper, item_mapper, user_inverse_mapper, item_inverse_mapper, user_ind, item_ind = utils.create_user_item_matrix(ratings)
		X_binary = X != 0

		grill_brush = "B00CFM0P7Y"
		grill_brush_ind = item_mapper[grill_brush]
		grill_brush_vec = X[:,grill_brush_ind]

		print(url_amazon % grill_brush)
		print(grill_brush_vec.shape)
		print(X.shape)

		# YOUR CODE HERE FOR Q1.2
		neigh = NearestNeighbors(n_neighbors=6)
		neigh.fit(X.T)
		near_ind = neigh.kneighbors(grill_brush_vec.T)[1][0]
		near_ind = near_ind[near_ind != grill_brush_ind]
		print("Nearest neighbours in Euclidean metric: ")
		for ind in near_ind:
			reviews = X_binary.getcol(ind).sum()
			print(item_inverse_mapper[ind] + " " + str(reviews))



		# YOUR CODE HERE FOR Q1.3
		neigh = NearestNeighbors(n_neighbors=6)
		neigh.fit(normalize(X.T))
		near_ind = neigh.kneighbors(normalize(grill_brush_vec.T))[1][0]
		near_ind = near_ind[near_ind != grill_brush_ind]
		print("Nearest neighbours in normalized Euclidean metric: ")
		for ind in near_ind:
			print(item_inverse_mapper[ind])


		neigh = NearestNeighbors(n_neighbors=6, metric = 'cosine')
		neigh.fit(X.T)
		near_ind = neigh.kneighbors(grill_brush_vec.T)[1][0]
		near_ind = near_ind[near_ind != grill_brush_ind]
		print("Nearest neighbours in cosine metric: ")
		for ind in near_ind:
			reviews = X_binary.getcol(ind).sum()
			print(item_inverse_mapper[ind] + " " + str(reviews))

	elif question == "3":
		data = load_dataset("outliersData.pkl")
		X = data['X']
		y = data['y']

		# Fit least-squares estimator
		model = linear_model.LeastSquares()
		model.fit(X,y)
		print(model.w)

		utils.test_and_plot(model,X,y,title="Least Squares",filename="least_squares_outliers.pdf")

	elif question == "3.1":
		data = load_dataset("outliersData.pkl")
		X = data['X']
		y = data['y']

		# YOUR CODE HERE

	elif question == "3.3":
		# loads the data in the form of dictionary
		data = load_dataset("outliersData.pkl")
		X = data['X']
		y = data['y']

		# Fit least-squares estimator
		model = linear_model.LinearModelGradient()
		model.fit(X,y)
		print(model.w)

		utils.test_and_plot(model,X,y,title="Robust (L1) Linear Regression",filename="least_squares_robust.pdf")

	elif question == "4":
		data = load_dataset("basisData.pkl")
		X = data['X']
		y = data['y']
		Xtest = data['Xtest']
		ytest = data['ytest']

		# Fit least-squares model
		model = linear_model.LeastSquares()
		model.fit(X,y)

		utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, no bias",filename="least_squares_no_bias.pdf")

	elif question == "4.1":
		data = load_dataset("basisData.pkl")
		X = data['X']
		y = data['y']
		Xtest = data['Xtest']
		ytest = data['ytest']

		model = linear_model.LeastSquaresBias()
		model.fit(X,y)

		utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares, with bias",filename="least_squares_with_bias.pdf")

	elif question == "4.2":
		data = load_dataset("basisData.pkl")
		X = data['X']
		y = data['y']
		Xtest = data['Xtest']
		ytest = data['ytest']

		for p in range(20):
			print("p=%d" % p)

			model = linear_model.LeastSquaresPoly(p)
			model.fit(X,y)

			utils.test_and_plot(model,X,y,Xtest,ytest,title="Least Squares with polynomial basis of degree " + str(p),filename="least_squares_with_polynomial_degree_" + str(p)+ ".pdf")

	else:
		print("Unknown question: %s" % question)

