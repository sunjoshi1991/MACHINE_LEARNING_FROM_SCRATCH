from __future__ import division , print_function
import numpy as np
import progressbar

from utils import train_test_split , standardize , normalize , to_catgeorical , mean_squared_error , accuracy_score , plot
from utils.misc import bar_widgets
from supervised_learning import XGBRegressor
from deep_learning.activation_functions import Sigmoid

class Logisticloss():

	def __init__(self):
		sigmoid = Sigmoid()
		self.log_function = sigmoid
		self.log_grad = sigmoid.gradient


	def loss(self,y , y_pred):
		y_pred = np.clip(y_pred , 1e-15 , 1-1e-15)
		p = self.log_function(y_pred)
		return y*np.log(p) + (1-y) * np.log(1-p)

	def gradient(self, y , y_pred):
		p = self.log_function(y_pred)
		return -(y-p)

	def hessian(self, y , y_pred):
		p = self.log_function(y_pred)

		return p * (1-p)


class XGBOOST(object):

	def __init__(slf, n_estimators=200 , learning_rate = 0.001 , min_smaples_split = 2 , min_impurity = 1e-7 , max_depth = 2):
		self.n_estimators = n_estimators
		self.learning_rate = learning_rate
		self.min_smaples_split  = min_smaples_split
		self.min_impurity = min_impurity
		self.max_depth = max_depth
		self.bar = progressbar.ProgressBar(widgets = bar_widgets)
		self.loss = Logisticloss()
		self.trees = []

		for _ in range(n_estimators):
			tree = XGBRegressor(min_smaples_split = self.min_smaples_split ,
				min_impurity = min_impurity,
				max_depth = self.max_depth,
				loss = self.loss)
			self.trees.append(tree)


	def fit(self, X, y):
		y = to_catgeorical(y)

		y_pred = np.zeros(np.shape(y))

		for i in self.bar(range(self.n_estimators)):
			tree = self.trees[i]
			y_and_pred = np.concatenate((y , y_pred) , axis = 1)
			tree.fit(X, y_and_pred)
			update_pred = tree.predict(X)

			y_pred = np.multiply(self.learning_rate , update_pred)


	def predict(self, X):
		y_pred = None

		for tree in self.trees:
			update_pred = tree.predict(X)
			if y_pred in None:
				y_pred = np.zeros_like(update_pred)

			y_pred-= np.multiply(self.learning_rate , update_pred)

		y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred) , axis=1 , keepdims =True)

		y_pred = np.argmax(y_pred , axis = 1)

		return y_pred






		

























