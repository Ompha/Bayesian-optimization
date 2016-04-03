from astropy.io import fits
from numpy import array
import numpy as np
import math as m

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm


from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
from numpy.random import uniform, seed

from scipy.interpolate import griddata
from scipy.integrate import quad
from scipy.optimize import leastsq
import scipy

hdulist = fits.open('data.fits')
image_data = hdulist[0].data
image_electron = 3* image_data
im_unflattened = image_electron

# define grid.
xi = np.arange(0, 256, 1)
yi = np.arange(0, 256, 1)
X, Y = meshgrid(xi, yi)

F_A= 25133.8303448
omega1 =  50.71626203  #fitted peak width
x0 = 125.9161215
y0 =  132.40840352
B = 400.04581579

F_B_1 =  7550.79980225
F_B_2 = 17953.06532653 
omega2 = 37.43214528
x0_b1 =  132.70626504
y0_b1 =  139.00606734
x0_b2 = 124.1091497 
y0_b2 = 130.64532061

# F, omega, x0, y0, B = [5.97528609e+05,    1.26493600e+05,  -3.27833897e+01, 7.74703938e+07,   3.99491522e+02]

def modelA(x, y):
	return F_A/(m.pi * 2 * omega1) * np.exp(-(x-x0)**2/(2 * omega1) - (y-y0)**2/(2 * omega1) ) + B

def modelB(x, y):
	return F_B_1/(m.pi * 2 * omega2) * np.exp(-(x-x0_b1)**2/(2 * omega2) - (y-y0_b1)**2/(2 * omega2) ) + F_B_2/(m.pi * 2 * omega2) * np.exp(-(x-x0_b2)**2/(2 * omega2) - (y-y0_b2)**2/(2 * omega2) ) + B

x_model = np.arange(0, 256, 1)
y_model = np.arange(0, 256, 1)

x = np.arange(0, 256, 1)
y = np.arange(0, 256, 1)

X, Y = meshgrid(x_model, y_model)

z_model0 = modelA(X,Y)
z_model = modelB(X,Y)
# z_model = z_model.flatten()


def grid(x, y, z, resX=256, resY=256):
    "Convert 3 column data to matplotlib grid"
    xi = linspace(min(x), max(x), resX)
    yi = linspace(min(y), max(y), resY)
    Z = griddata(x, y, z, xi, yi)
    X, Y = meshgrid(xi, yi)
    return X, Y, Z

#Model A calculation from HW3 
x  = np.arange(0, 256, 1)
y  = np.arange(0, 256, 1)
image_electron = image_electron.flatten()

X = X.flatten()
Y = Y.flatten()

def functions_chiA(p):
	F_A, omega1, x0_a, y0_a, B = p

	z_model = F_A/(m.pi * 2 * omega1) * np.exp(-(X-x0_a)**2/(2 * omega1) - (Y-y0_a)**2/(2 * omega1) ) + B
	z_model = z_model.flatten()
	error = image_electron - z_model
	return error

sig_est = (image_electron/3).flatten()

def functions_chiA_err(p):
	F_A, omega1, x0_a, y0_a, B = p

	z_model = F_A/(m.pi * 2 * omega1) * np.exp(-(X-x0_a)**2/(2 * omega1) - (Y-y0_a)**2/(2 * omega1) ) + B
	z_model = z_model.flatten()
	error = (image_electron - z_model)
	error = np.divide(error, sig_est)
	return error

def functions_chi(p):
	F_B_1, F_B_2, omega2, x0_b1, y0_b1, x0_b2, y0_b2 = p

	z_model = F_B_1/(m.pi * 2 * omega2) * np.exp(-(X-x0_b1)**2/(2 * omega2) - (Y-y0_b1)**2/(2 * omega2) ) + F_B_2/(m.pi * 2 * omega2) * np.exp(-(X-x0_b2)**2/(2 * omega2) - (Y-y0_b2)**2/(2 * omega2) ) + B

	z_model = z_model.flatten()
	error = image_electron - z_model
	return error

def functions_chi_err(p):
	F_B_1, F_B_2, omega2, x0_b1, y0_b1, x0_b2, y0_b2 = p

	z_model = F_B_1/(m.pi * 2 * omega2) * np.exp(-(X-x0_b1)**2/(2 * omega2) - (Y-y0_b1)**2/(2 * omega2) ) + F_B_2/(m.pi * 2 * omega2) * np.exp(-(X-x0_b2)**2/(2 * omega2) - (Y-y0_b2)**2/(2 * omega2) ) + B

	z_model = z_model.flatten()
	error = (image_electron - z_model)/sig_est
	return error

start_cond_p0A = [52700, (m.sqrt(3)*10)**2, 123, 123, 0.5]

resultsA = leastsq(functions_chiA, start_cond_p0A, full_output=1)
bestfitA = resultsA[0]
Cov_A = resultsA[1]

#Plots
plt.figure()
# X,Y,Z = grid(x, y, z_model, resX=256, resY=256)
# CS = plt.contour(X,Y,Z)
plt.subplot(221)
plt.imshow(im_unflattened)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Raw Data From Measurement')

plt.subplot(222)
plt.imshow(im_unflattened - z_model0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Separated Background Noise')

plt.subplot(223)
plt.imshow(z_model0)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Raw Data as Single Star With Gaussian Distribution')

plt.subplot(224)
plt.imshow(z_model)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Raw Data as Binary Star with Bimodal Guassian')
figname = 'Star Modeling with Gaussain Distribution.jpg'
plt.savefig(figname, format='png')
plt.show()























