from __future__ import print_function , division
import numpy as np
from utils import accuracy_score
from activation_fucntions import Sigmoid


## Gradient of any fucntion is first order derivative of that fucntion

class Loss(object):
	def loss(self, y_true, y_pred):
		return NotImplementedError()

	def gradient(self, y,y_pred):
		return NotImplementedError()

	def acc(self, y , y_pred):
		return 0

### Regression loss -->> for contionous outcome

class SquareLoss(Loss):
	def __init__(self):
		pass

	def loss(self, y , y_pred):
		return 0.5 * np.power((y-y_pred) , 2)   #### eqn (1)

	def gradient(self, y , y_pred):
		return -(y-y_pred)              ### derivative of  eqn(1)

### For CLassifictions, cross entrpoy is similiar to log loss for binary

class CrossEntropy(Loss):
	def __init__(self):
		pass

	def loss(self,y,p):
		p = np.clip(p, 1e-15 , 1- 1e-15)
		return -y * np,log(p) - (1-y) * np.log(1-p)

	def acc(self, y, p):
		return accuracy_score(np.argmax(y, axis=1) , np.argmax(p, axis=1))

	def gradient(self, y , p):
		return -(y/p) + (1-y)/(1-p)




