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
		self.v = self.momentum*self.v-self.lr*grads
		params += self.v