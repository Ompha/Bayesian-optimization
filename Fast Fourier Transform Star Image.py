# FFT Based Convolution for 3D Galaxy Emission with Point spread Function
# AUTHOR: Cicero Xinyu Lu
# Date: 03/01/2016

import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from numpy import array, linspace, meshgrid
from numpy.random import uniform, seed
import numpy as np
import math as m
import scipy as sp 

from mpl_toolkits.mplot3d import axes3d
import scipy
from scipy.interpolate import griddata
from scipy.optimize import leastsq
from scipy.integrate import quad
import scipy.special as special
from astropy.io import fits

#Problem 1
#Input data points
with open('hw5-hkk-trials.txt') as f:
	lines = (line for line in f if not line.startswith('#'))
	raw_data = np.loadtxt(lines)

# Define the probability distributions
	# 0th order modified Bessel function 
		#scipy.sepcial.jv(n, z)
def I0(z):
	result = special.iv(0,z)
	return result

	# prob function
def probability(alpha, beta, x):
	if x >=0:
		p_x = alpha**(-1) *np.exp(-(x + beta)/alpha ) * I0(2.0 * np.sqrt(x * beta)/alpha)
		return p_x
	else:
		return 0 

# Define the test range for alpha and beta
alpha_trail = np.linspace(0.1, 100, 200)
beta_trail = np.linspace(0.1, 150, 200)

#Use Least Square method to get the value for alpha and beta
def total_prob(alpha, beta):
	err = 1
	for i in range(50):
		err = err * probability(alpha, beta, raw_data[i])
	return err

results = total_prob(alpha_trail[:,None], beta_trail[None, :])

plt.figure()
plt.imshow(results)
plt.suptitle('Posterior density function using Modified Rice Distribution')
figname = 'Problem 1(a).jpg'
plt.savefig(figname, format='png')
plt.show()

print "The maximum likelihood is: ", np.nanmax(results)
print "is at index: ", np.where(results == np.nanmax(results))
print "Alpha , Beta is: ", alpha_trail[92], beta_trail[33]

# Thus : 
# Alpha , Beta is:  46.2849246231 24.9577889447

# Now try Guassian:
def gaussian_model(x):
	mean = np.sum(raw_data)/50
	sigma = np.sqrt(mean)
	return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x- mean)**2/(2* sigma**2))
for i in range(50):
	prob = 0
	prob = prob * gaussian_model(raw_data[i])

print "prob for gaussian model is: ", prob
print "the likelihood ratio is ", total_prob(alpha_trail[92], beta_trail[33])/prob

x_axis_for_plot = np.arange(0, 50, 1)
new_test = np.sort(raw_data)
new_y = []

for i in range(50):
	new_y.append(gaussian_model(new_test[i]))
	
plt.figure()
plt.plot(x_axis_for_plot, new_y)
plt.show()


#=============================================#
#=============================================#
#=============================================#
#Problem 2 
#Import the data:from astropy.io import fits
hdulist = fits.open('hw6prob2_model.fits')
image_data = hdulist[0].data
#Fits file handling
#HDUList is a list-like collection of HDU(Header Data Unit) objects, Consist a header and a data array
# hdulist.info() summerizes the content of fits file
# hdulist[0].header[number] returns a 

image_data_integrated = [[0 for x in range(128)] for x in range(128)]

for i in range(0,128):
	image_data_integrated = np.add(image_data_integrated, image_data[i])

#Check the dimension of the newly integrated array
print "Size of the integrated array is ", np.shape(image_data_integrated) 

plt.figure(1)
plt.imshow(image_data_integrated)
plt.show()

# Now, convolve with PSF for simulation
hdulist = fits.open('hw6prob2_psf.fits')
psf_data = hdulist[0].data
#We've checked that the psf size is (64, 64), thus we need to zero-pad it to the size of (256, 256). Same padding for image_data

# Use Numpy.pad:
# np.lib.pad(array, pad_width, **kwargs)
def padwithzero(vector, pad_width, iaxis, kwargs):
	vector[:pad_width[0]] = 0
	vector[-pad_width[1]:] = 0
	return vector

psf_data_padded = np.lib.pad(psf_data, 96, padwithzero)
image_data_padded = np.lib.pad(image_data_integrated, 64, padwithzero)
#Check if it's padded correctly using imshow
plt.figure(2)

plt.subplot(2,2,1)
plt.imshow(image_data_integrated)
plt.subplot(2,2,2)
plt.imshow(image_data_padded)
plt.subplot(2,2,3)
plt.imshow(psf_data)
plt.subplot(2,2,4)
plt.imshow(psf_data_padded)
plt.suptitle('Problem 2: Original(left) and Zero padded(right) Images')
figname = 'Problem 2(a).jpg'
plt.savefig(figname, format='png')
plt.show()

#Padding seems okay,go on

# FFT data and psf
fft_obs = np.fft.fft2(image_data_padded)
fft_psf = np.fft.fft2(psf_data_padded)

#Shift the image by N/2
fft_obs = np.fft.fftshift(fft_obs)
fft_psf = np.fft.fftshift(fft_psf)

plt.figure(3)
plt.subplot(1,2,1)
plt.imshow(abs(fft_obs))
plt.subplot(1,2,2)
plt.imshow(abs(fft_psf))

plt.suptitle('Problem 2: Data and Psf after FFT')
figname = 'Problem 2(b).jpg'
plt.savefig(figname, format='png')
plt.show()

sum_matrix = np.multiply(fft_obs, fft_psf)
plt.figure(4)
plt.imshow(abs(sum_matrix))
plt.show()

# real_scene = np.fft.ifft2(sum_matrix)
# real_scene = np.fft.ifftshift(real_scene)

# plt.figure(5)
# plt.imshow(np.abs(real_scene))
# plt.suptitle('Problem 2: Simulated Observation with FFT')
# figname = 'Problem 2(c).jpg'
# plt.savefig(figname, format='png')
# plt.show()
