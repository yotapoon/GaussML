import numpy as np
import matplotlib.pyplot as plt
from pylab import *
np.set_printoptions (precision=4,suppress=True)

def lm (X):
  N = len(X)
  y = X[:,0]
  X = np.vstack([np.ones(N),X[:,1]]).T
  w = inv(X.T.dot(X)).dot(X.T).dot(y)
  return w

def simple_plot (X):
  xmin,xmax = -5,5
  ymin,ymax = -5,5
  plt.scatter(X[:,1],X[:,0],marker='x',s=80)
  plt.plot([xmin,xmax],[0,0],'k',linewidth=1)
  plt.plot([0,0],[ymin,ymax],'k',linewidth=1)
  plt.axis([xmin,xmax,ymin,ymax])

def lm_plot (X):
  xmin,xmax = -5,5
  ymin,ymax = -5,5
  simple_plot (X)
  w = lm (X)
  M = 20
  xx = linspace(xmin,xmax,M)
  yy =[w[0]+w[1]*x for x in xx]
  plt.plot (xx,yy)
  gca().set_aspect(1)
  print (w)

def lm_predict (x,w):
  yhat = w[0]+w[1]*x
  return yhat
def lm_error (x,y,w):
  yhat = lm_predict (x,w)
  return (y - yhat)**2
def lm_errors (X,w):
  return [lm_error(x[0],x[1],w) for x in X]
def lm_mse (X,w):
  errors = lm_errors(X,w)
  print (mean(errors))
