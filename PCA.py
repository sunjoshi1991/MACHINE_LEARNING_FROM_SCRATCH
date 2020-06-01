from __future__ import  print_function, division
import numpy as np
from utils import calculate_covariance_matrix


## PCA class

class PCA():

	def transform(self,X,n_components):
		"""
		fit dataset to principal_components and return transformed dataset

		"""
		## get eigen vals and vectors from covar matrix
		covariance_matrix = calculate_covariance_matrix(X)
		eigenvalues , eigenvectors = np.linalg.eig(covariance_matrix)

		## sort eigenvalues to get largest
		idx = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[idx][:n_components]
		eigenvectors = np.atleast_1d(eigenvectors[]:,idx)[:n_components]

		## project data onto principal components

		X_transformed = X.dot(eigenvectors)

		return X_transformed