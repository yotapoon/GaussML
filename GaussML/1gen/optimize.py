class optimize:
	def __init__(self,params):
		self.params = params
		self.lr = lr
		
	def update(self,params,grads):
		for key in params.keys():
			params[key] -= self.lr * grads[key]
	