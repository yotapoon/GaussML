import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train,t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784,hidden_size=50,output_size=10)

for i in range(iters_num):
	print(i)
	batch_mask = np.random.choice(train_size,batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	
	grads = network.gradient(x_batch,t_batch)
	
	optimizer.update_dict(network.params,grads)
	
	loss = network.loss(x_batch,t_batch)
	train_loss_list.append(loss)
