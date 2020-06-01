import numpy as np

## sigmoid activation fucntion

class Sigmoid():
	def _call__ (self,x):
		return 1/1+np.exp(-x)

	def gradient(self,x):
		return self._call__(x) * (1-self._call__(x))


### Softmax 

class Softmax():
	def __call__ (self,x);
		e_X = np.exp(x-np.max(x,axis =-1, keepdims = True))
	return e_X/np.sum(e_X, axis = 1 , keepdims = True)

	def gradient(self, x):
		p = self.__call__(x)
		return p *(1-p)

class Tanh():
	def __ call__(self,x):
		return 2/(1+np.exp(-2*x))-1

	def gradient(self,x):
		return 1- np.power(self.__call__(x),2)


### relu is the most optimized and faster compared to others

class Relu():
	def __call__(self,x):
		return np.where(x>=0 , x, 0)

	def gradient(self,x):
		return np.where(x>=0 , 1, 0)

## variation in Relu ->> Leaky Relu

class LeakyRelu():
	def __init__ (self,alpha=0.2):
		self.alpha = alpha

	def __call__ (self,x):
		return np.where(x >=0 , x , self.alpha *x)

	def gradient(self,x):
		return np.where(x>=0 , 1, self.alpha)




