 ### Utility fucntions for Kernels , data preocessing and performance metrics


 ## Kernels
 import numpy as np


 def linear_kernel(**kwargs):
 	def f(a,b):
 		return np.inner(a,b)
 	return f

 def polynomial_kernel(power , coef, **kwargs):
 	def f(a,b):
 		return (np.inner(a,b) + coef) ** power
 	return f

 def RBF(gamma , **kwargs):
 	def f(a,b):
 		distance = np.linalg.norm(a-b) **2
 		return np.exp(-gamma * distance)
 	return f


 ### data preproessing 

 from __future__ import division
 from itertools import combinations_with_replacement
 import numpy as np
 import math , sys

 def shuffle_data(X,y, seed = None): ## random shuffle data
 	if seed:
 		np.random.seed(seed)
 	idx = np.arange(X.shape[0])
 	np.random.shuffle(idx)

 	return X[idx], y[idx]

 def normalize(X, axis =1, order=2):
 	l2 = np.atleast_1d(np.linalg.norm(X. order, axis))
 	l2[l2==0] =1

 	return X/np.expand_dims(l2, axis=1)

 def standardize(X):
 	X_std = X
 	mean = X.mean(axis=0)
 	std = X.std(axis = 0)
 	for col in range(np.shape(X)[1]):
 		if std[col]:
 			X_std[:, col] = (X_std[:,col]-mean[col]/std[col])
 	return X_std

 def train_test_split(X, y , test_size = 0.3 , shuffle = True, seed = None):
 	if shuffle:
 		X, y = shuffle_data(X,y ,seed)

 	split_i = len(y) -int(len(y))//(1/test_size)
 	X_train , X_test = X[:split_i], X[split_i:]
 	y_train , y_test = y[:split_i] , y[split_i]

 	return X_train , X_test , y_train , y_test

 def k_fold_cross_validation(X,y, k, shuffle = True):
 	if shuffle:
 		X, y = shuffle_data(X,y)
 	n_samples = len(y)


 def to_categorical(x, n_col = None):  ### one hot encoding
 	if not n_col:
 		n_col = np.amax(x) +1
 	one_hot = np.zeros(x.shape[0] , n_col)
 	one_hot[np.arange(x.shape[0]) , x]=1
 	return one_hot

 def to_nominal(x):
 	return np.argmax(x, axis = 1)

 def make_diagonal(x):
 	m = np.zeros((len(x), len(x)))
 	for i in range(len(m[0])):
 		m[i,i ] = x[i]
 	return m



### Performance metrics

def mean_squared_error(y_true , y_pred):
	mse = np.mean(np.power(y_true - y_pred))
	return mse

def accuracy_score(y_true , y_pred):
	accuracy = np.sum(y_true==y_pred, axis=0) / len(y_true)
	return accuracy












































































































































































































































