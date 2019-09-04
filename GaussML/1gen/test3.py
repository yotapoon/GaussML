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

class SGD:
	def __init__(self,lr = 0.01):
		self.lr = lr
		
	def update_dict(self,params,grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key]
	
	def update_array(self,params,grads):
		for i in range(params.size):
			params[i] -= self.lr*grads[i]
def f(x):
	return 0.05*x[0]**2+x[1]**2
x = np.array([-7.0,2.0])
T = 100
trajectory_x = []
trajectory_y = []
optimizer = AdaGrad(0.7)
for t in range(T):
	g = numerical_gradient(f,x)
	trajectory_x.append(x[0])
	trajectory_y.append(x[1])
	optimizer.update_array(x,g)
plt.plot(trajectory_x,trajectory_y)