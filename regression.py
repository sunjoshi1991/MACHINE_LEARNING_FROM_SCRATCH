from __future__ import print_function, division
import numpy as np
import math
from mlfromscratch.utils import normalize, polynomial_features


class l1_reg():

	def __init__(self, alpha):
		self.alpha = alpha

		def __call__(self, w):
			return self.alpha * np.linalg.norm(w)

		def grad(self, w):
			return self.alpha * np.sign(w)

class l2_reg():

	def __init__(self, alpha):
		self.alpha = alpha

	def __call__(self,w):
		return self.alpha * 0.5 * w.T.dot(w)

	def grad(self, w):
		return self.alpha *w

class l1_l2_reg():

	def __init__(self, alpha, l1_ratio = 0.5):
		self.alpha = alpha
		self.l1_ratio = l1_ratio

	def __call__(self, w):
		l1_contr = self.l1_ratio * np.linalg.norm(w)
		l2_contr = (1-self.l1_ratio) * 0.5 * w.T.dot(w)
		return self.alpha *(l1_contr + l2_contr)

	def grad(self,w):
		l1_contr = self.l1_ratio * np.sign(w)
		l2_contr = (1-self.l1_ratio)*w

		return self.alpha * (l1_contr + l2_contr)

