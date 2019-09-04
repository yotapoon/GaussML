import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

def sigmoid(x):
	return 1/(1+np.exp(-x))

x = np.random.randn(1000,100)#1000datas size 100
node_num = 100
hidden_layer_size = 5
activations ={}
weight_std = 0.01

for i in range(hidden_layer_size):#i:label of hidden_layer
	if i != 0:
		x = activations[i-1]
	
	w = np.random.randn(node_num,node_num)*weight_std
	
	z = np.dot(x,w)
	a = sigmoid(z)
	activations[i] = a

for i,a in activations.items():
	plt.subplot(1,len(activations),i+1)
	plt.title(str(i+1)+"-layer")
	plt.hist(a.flatten(),30,range=(0,1))
plt.show()
plt.savefig("histogram2.png")