#2D Gaussian Optimzation Countour
#AUTHOR: Cicero Xinyu Lu
# Date: 1/21/2016

from numpy import array
import math as m
import numpy as np
import scipy.ndimage as ni

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import cm


# Initial Data
# A set of Measurements {y} at known location {x}
# Fit a line (y = a0 + a1 * x) with maximun likelihood technique
array_X = ([3, 8, 9])
array_Y = ([8.95, 7.06, 9.18])

# Given Gaussian Uncertainty
N_dim = int(len(array_X))
sum_result = 0 
sigma = 1.0


# Define the Likelihood function from Gaussian Distribution
# Gaussian ~ exp( -( y - (a0 + a1 * x) )^2/ 2sigma^2)
# 70 percent of time, measurement is reliable

def partial_likelihood(a0, a1, ii):
	return (0.7 * m.exp(-(array_Y[ii] - a0 - a1 * array_X[ii])**2 /(2* sigma**2)) + 0.3 * m.exp(-(array_Y[ii] - a0 - a1 * array_X[ii]- 5)**2 /(2* sigma**2))) 

def fcn_likelihood(a0, a1):
	return partial_likelihood(a0, a1, 0)* partial_likelihood(a0, a1, 1) * partial_likelihood(a0, a1, 2)

a0_array = np.arange(-3, 5, 0.05)
a1_array = np.arange(0, 1.5,  0.009375)
likelihood_array = []




len_array = int (len(a0_array))

vector_plot = ([])
for i in range(0, 160):
	for j in range(0, 160):
		vector_plot.append((a0_array[i],a1_array[j],fcn_likelihood(a0_array[i],a1_array[j])))

from numpy import linspace, meshgrid
from matplotlib.mlab import griddata

def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = meshgrid(xi, yi)
    return X, Y, Z

x, y, z = zip(*vector_plot)
index_z = z.index(max(z))
print "a0", x[index_z], "a1", y[index_z], "Cauchy Distribution", z[index_z]

# Plot Contour
fig = plt.figure()
X,Y,Z = grid(x, y, z, resX=100, resY=100)
ax = fig.add_subplot(1,2,1)
CS = plt.contour(X,Y,Z)
plt.xlabel('a0')
plt.ylabel('a1')
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Contour Plot: Optimizing a0 and a1')

# surface_plot with color grading and color bar
ax = fig.add_subplot(1,2, 2, projection = '3d')
#ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
plot2 = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, cmap=cm.coolwarm, linewidth=0, antialiased=False)
cb = fig.colorbar(plot2, shrink=0.5)
plt.title('3D Plot of Parameter space a0 and a1')
figname = 'Gaussian Optimization Plots.png'
plt.savefig(figname, format='png')

plt.show()
