from __future__ import division , print_function
import numpy as np
import cvxopt

from utils import train_test_split , normalize , accuracy_score
from utils.kernels import *
from utils import Plot



## create SVM class

class SupportVectorMachines (object):

	def __init__(self, C=1 , kernel = rbf_kernel , power = 4, gamma = None , coef =	4):
		self.C = C
		self.kernel =kernel
		self.power= power
		self.gamma = gamma
		self.coef = coef
		self.lagr_multipliers = None
		self.support_vectors = None
		self.support_vectors_labels = None
		self.intercept = None


	def fit(f , X , y):
		n_samples , n_features = np.shape(X)

		if not self.gamma:
			self.gamma = 1/n_features

		self.kernel = self.kernel(power =self.power ,gamma = self.gamma , coef = self.coef)

		kernel_matrix = np.zeros((n_samples , n_features))
		for i in range(n_samples):
			for j in range(n_samples):
				kernel_matrix[i,j] = self.kernel(X[i] , X[j])

		## qudratic optimizer
		P = cvxopt.matrix(np.outer(y,y) * kernel_matrix, tc = 'd')
		q = cvxopt.matrix(np.ones(n_samples) * -1)
		A = cvxopt.matrix(y,(1,n_samples) , tc = 'd')
		b = cvxopt.matrix(0, tc = 'd')

		if not self.C:
			G = cvxopt.matrix(np.identity(n_samples) * -1)
			h = cvxopt.matrix(np.zeros(n_samples))
		else:
			G_max = np.identity(n_samples) * -1
			G_min = np.identity(n_samples)
			G = cvxopt.matrix(np.vstack((G_max, G_min)))
			h_max = cvxopt.matrix(np.zeros(n_samples))
			h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
			h = cvxopt.matrix(np.vstack((h_max,h_min)))

		## solve the quaddrattic equation
		minimal = cvxopt.solvers.qp(P,q,G,h,A,b)

		## lagrangian multipliers

		lagr_multiply = np.ravel(minimal['X'])

		## Support Vectors

		## get non zero lagrangian indexes and corresponding multipliers
		idx = lagr_multiply > 1e-7
		self.lagr_multipliers = lagr_multiply[idx]
		## get support vectors and  lables
		self.support_vectors = X[idx]
		self.support_vectors_labels = y[idx]

		## calculate b(intercept) for 1st support vector

		self.intercept = self.support_vectors_labels[0]
		for i in range(len(self.lagr_multipliers)):
			self.intercept -= self.lagr_multipliers[i] * self.support_vectors_labels[i] * self.kernel(self.support_vectors[i], self.support_vectors[0])


	def predcit(self,X):
		y_pred = []
		for sample in X:
			prediction = 0
			for i in range(len(self.lagr_multipliers)):
				prediction += self.lagr_multipliers[i] * self.support_vectors_labels[i] * self.kernel(self.support_vectors[i] , sample)
				prediction += self.intercept
				y_pred.append(np.sign(prediction))
		return np.array(y_pred)

















































		