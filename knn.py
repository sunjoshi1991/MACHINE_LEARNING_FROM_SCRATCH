from __future__ import print_function , division
from utils import euclidean_distance
import numpy as np

## KNN class

class KNN():

	def __init__(self,k=5):
		self.k = k

	def vote(self , neighbors):
		counts = np.bincount(neighbors.astyoe('int'))

		return counts.argmax()

	def predict(self, X_test , X_train, y_train):
		y_pred = np.empty(X_test.shape[0])

		for idx , test_sample in enumerate(X_test):

			index = np.argsort([euclidean_distance(test_sample , x) for x in X_train])[:self.k]

			k_nearest_neighbors = np.array([y_train[i] for i in index])

			y_pred[i] = self.vote(k_nearest_neighbors)

		return y_pred
