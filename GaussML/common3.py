import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from numpy.random import multivariate_normal as mvnrand
from numpy import exp,sqrt
from numpy.linalg import inv,norm


def kgauss(params):
	[tau,sigma] = params
	return lambda x,y: params[0]*exp(-(x-y)**2/(2*params[1]*params[1]))
	
def kv(x,xtrain,kernel):
	return np.array([kernel(x,xi) for xi in xtrain])
	
def kernel_matrix(xx,kernel):
	eta = 0.1
	N = len(xx)
	return np.array( [kernel (xi,xj) for xi in xx for xj in xx] ).reshape(N,N)+eta*np.eye(N)
	
def gpr(xx,xtrain,ytrain,kernel):
	K = kernel_matrix(xtrain,kernel)
	Kinv =inv(K)
	ypr = [];spr = []
	for x in xx:
		s = kernel(x,x)+eta
		k = kv(x,xtrain,kernel)
		ypr.append(k.T.dot(Kinv).dot(ytrain))
		spr.append(s-k.T.dot(Kinv).dot(k))
	return ypr,spr
	
def gpplot(xx,xtrain,ytrain,kernel,params):
	ypr,spr = gpr(xx,xtrain,ytrain,kernel(params))
	plot(xtrain,ytrain,"bx",markersize = 16)
	plot(xx,ypr,"b-")
	fill_between(xx,ypr-2*sqrt(spr),ypr+2*sqrt(spr),color="#ccccff")
	

# plot parameters
N    = 100
xmin = -1
xmax = 3.5
ymin = -1
ymax = 3

# GP kernel parameters
eta   = 0.1
tau   = 1
sigma = 1

train = np.loadtxt ('gpr.dat',dtype=float)

xtrain = train.T[0]
ytrain = train.T[1]
kernel = kgauss
params = [tau,sigma]
xx = np.linspace (xmin, xmax, N)
gpplot (xx, xtrain, ytrain, kernel, params)