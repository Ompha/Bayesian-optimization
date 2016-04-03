#Integrating Poisson Distribution
#AUTHOR Cicero Lu
#Date 01/14/2016
import math

import scipy.special
import scipy.stats
from scipy.integrate import quad

import numpy as np
from numpy import array
from numpy import inf

import matplotlib.pyplot as plt

# array1 = []
# array2 = []

x1 = np.arange(0, 11, 1)
x2 = np.arange(0, 11, 1)

def integrand(l, m, n):
	return l**(m+n) * np.exp(-2*l)/math.factorial(m)/math.factorial(n)

def expint(m, n):
    return quad(integrand, 0, inf, args=(m, n))[0]

vec_expint = np.vectorize(expint)

array1 = vec_expint(x1, 3)
array2 = vec_expint(x2, 6)


fig = plt.figure(1)
ax1 = plt.subplot(211)
plt.plot(x1, array1, '-bo')
plt.xlabel('M (photons)')
# ax1.set_xticks(np.arange(100, 900, 0.25))
plt.ylabel('P(M | 3)')

ax2 = plt.subplot(212)
plt.plot(x2, array2, '-ro')
plt.xlabel('M (photons)')
# ax1.set_xticks(np.arange(100, 900, 0.25))
plt.ylabel('P(M | 6)')

fig.suptitle('Figure: Probability P(M | N) at M = 3, 6 where M ranges [0, 10]')
figname = 'plot2_hw1.jpg'
plt.savefig(figname, format='png')
plt.show()
