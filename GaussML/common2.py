import numpy as np
import matplotlib.pyplot as plt
from pylab import *
np.set_printoptions (precision=4,suppress=True)

def gaussianKernel( x1, x2 ):
  a = 0.2
  return np.exp( - a * (x1 - x2) * (x1 - x2) )


def computeGram(elements, k):
    n    = len(elements)
    gram = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1):
            gram[i, j] = k(elements[i], elements[j])

    upTriIdxs       = np.triu_indices(n)
    gram[upTriIdxs] = gram.T[upTriIdxs]

    return gram