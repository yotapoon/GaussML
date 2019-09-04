class SGD:
	def __init__(self,lr = 0.01):
		self.lr = lr
		
	def update_dict(self,params,grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key]
	
	def update_array(self,params,grads):
		for i in range(params.size):
			params[i] -= self.lr*grads[i]
			