import math 
import numpy as np
from __future__ import print_function , division
from mlfromscratch.utils import normalize, polynomial_features

class LogisticRegression():

	def __init__(self, learning_rate= 0.1 , gradient_descent = True):
		self.param = None
		self.learning_rate = learning_rate
		self.gradient_descent = gradient_descent
		self.sigmoid = sigmoid


	def init_param(self, X):
		num_features = np.shape(X)[1]
		limit = 1/math.sqrt(num_features) ## init wit [-1/sqrt(N) , 1/sqrt(N)]
		self.param = np.random.uniform(-limit , limit , (num_features,))

	def fit(self, X, y , num_iter = 1000):
		self.init_param(X)

		for i in range(num_iter):   ## tune params 

			## make new predictions
			y_pred = self.sigmoid(X.dot(self.param))
			if self.gradient_descent:  ## move the gradient of loss fucntion w.r.t params to minimize loss
				self.param -= self.learning_rate * -(y- y_pred).dot(X)
			else:
				diag_grad = make_diagonal((self.sigmoid.gradient(X.dot(self.param))))   ## make diagonal matrix fo sigmoid fucntion column vector

				self.param = np.linalg.pinv(X.T.dot(diag_grad).dot(X).dot(X.T).dot(diag_grad.dot(X).dot(self.param) + y - y_pred))  ## batch optimization

	def predict(self, X):
		y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
		return y_pred


