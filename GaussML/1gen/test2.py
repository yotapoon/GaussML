import numpy as np
import matplotlib.pyplot as plt
def numerical_gradient(f,x):
	h = 1e-4
	grad = np.zeros_like(x)
	
	for idx in range(x.size):
		tmp = x[idx]
		x[idx] = tmp+h
		fxh1 = f(x)
		
		x[idx] = tmp-h
		fxh2 = f(x)
		
		grad[idx] = (fxh1-fxh2)/(2*h)
		x[idx] = tmp
		
	return grad

class Momentum:
	def __init__(self,lr = 0.01,momentum = 0.9):
		self.lr = lr
		self.momentum = momentum
		self.v = None
	
	def update_dict(self,params,grads):
		if self.v is None:
			self.v = {}
			for key,val in params.items():
				self.v[key] = np.zeros_like(val)
				
		
		for key in params.keys():
			self.v[key] = self.momentum*self.v[key]-self.lr*grads[key]
			params[key] += self.v[key]
	
	def update_array(self,params,grads):
		if self.v is None:
			self.v = np.zeros_like(params)
		for i in range(params.size):
			self.v[i] = self.momentum*self.v[i]-self.lr*grads[i]
			params[i] += self.v[i]

def f(x):
	return 0.05*x[0]**2+x[1]**2
x = np.array([-7.0,2.0])
T = 100
trajectory_x = []
trajectory_y = []
optimizer = Momentum(0.7,0.9)
for t in range(T):
	g = numerical_gradient(f,x)
	trajectory_x.append(x[0])
	trajectory_y.append(x[1])
	optimizer.update_array(x,g)
plt.plot(trajectory_x,trajectory_y)