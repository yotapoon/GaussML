import numpy as np
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

def function_2(x):
	return np.sum(x**2)

print(numerical_gradient(function_2,np.array([3.0,4.0])))