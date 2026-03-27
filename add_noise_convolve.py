# Author: Elouan Spaan
# Adds noise to a synthetic cube and convolved with a beam of choice.
#

import astropy as ap
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from numpy.random import Generator, PCG64  # what is this PGC64?


# constants
AU_cgs = 1.496e13 # AU in cm

# model and beam settings
pixsize_cm = 1.2341332017491162e13 # cm
bmaj = 0.15 # FWHM beam major, arcsec
bmin = 0.15 # FWHM beam minor, arcsec

# source settings
d_pc = 150. # distance to source, pc
PA = -90 # position angle, deg

# other
noise_std = 5.4e-6 # Jy/pix area, taken from J1604 fiducial data and corrected for beam and pixel size (but it really doesn't matter that much)
noise_std = 5.4e-5 # new noise level to test

# functions
def pix_cm_to_arcsec(pixsize_cm, d_pc):
    """
    Convert pixel in cm to pixel in arcsec
    
    :param pix_cm: pixel size in cm
    :param d: distance to the source in parsec
    """
    return (pixsize_cm/AU_cgs) / d_pc

def beam_to_arcsec_pix(bmaj, bmin, pixsize_cm, d_pc):
    # Compute beam area in square arcsec and square pix

    beam_arcsec = (np.pi*bmaj*bmin)/(4*np.log(2)) # conversion factor beam to arcsec area - divide beam by this
    beam_pix = beam_arcsec / (pix_cm_to_arcsec(pixsize_cm, d_pc))**2 # convert beam to pix area - divide beam by this; assumes square pixels!
    return beam_arcsec, beam_pix

def JyPix_to_JyBeam(data, pixsize_cm, d_pc):
    # convert intensity from Jy/pix to Jy/beam

    flux_pix = data
    pix_arcsec = pix_cm_to_arcsec(pixsize_cm, d_pc)
    _, conv_factor_pix = beam_to_arcsec_pix(bmaj, bmin, pixsize_cm, d_pc)
    flux_beam = flux_pix * conv_factor_pix # convert pixel flux to beam flux

    return flux_beam

def add_noise(img):
    # add white noise to an image
    # returns the Gaussian noise profile + noisy image

    noisy_img = np.zeros(img.shape, np.float32) # new noisy img
    pixsize_arcsec = pix_cm_to_arcsec(pixsize_cm, d_pc)
    #print("Pixel size in arcsec: ", pixsize_arcsec)
    
    rng = Generator(PCG64()) # proper way to generate random numbers; can also use np.random.default_rng()
    gaussian = noise_std * rng.standard_normal((img.shape[0], img.shape[1], img.shape[2])) # draw samples from a standard normal distribution (mean = 0, stdev = 0.0054 with mean 23 mJy)

    # add to data
    noisy_img = img + gaussian

    return gaussian, noisy_img

def convolve_img(img, squarepix = True, bmin = bmin, bmaj = bmaj):
    """
    Convolve image with a beam.

    To-do:
    * Add pix size and distance as cusomisable parameters

    """

    # compute standard deviation of beam PSF from FWHM
    pix_arcsec = (pixsize_cm/AU_cgs) / d_pc # pixel in arcsec, assumes square pix
    sigma_x = bmaj / (2 * np.sqrt(2 * np.log(2))) / pix_arcsec  # get std from FWHM and convert to pixel units
    sigma_y = bmin / (2 * np.sqrt(2 * np.log(2))) / pix_arcsec

    stdev_x = sigma_x
    stdev_y = sigma_y

    print(f"sigma_x: {sigma_x}")
    print(f"sigma_y: {sigma_y}")

    if squarepix == True:
        stdev = sigma_x # in case of square pixels
    else: 
        print("Pixels not square!")
    
    # conversion factor Jy/pix to Jy/beam
    bmaj *= u.arcsec
    bmin *= u.arcsec
    cdelt1 = (pixsize_cm * u.cm / u.au / d_pc).cgs * u.arcsec
    cdelt2 = (pixsize_cm * u.cm / u.au / d_pc).cgs * u.arcsec
    conversion_factor = (np.pi * bmaj * bmin / np.abs(cdelt1) / np.abs(cdelt2) / (4 * np.log(2))).cgs.value # Jy/beam to Jy/arcsec
    print("/pix to /arcsec conversion factor: ", conversion_factor)

    g2D_kernel_J1604 = Gaussian2DKernel(stdev, theta = PA) # J1604 data beam std at d = 150.2 pc, FWHM (arcsec) = 0.15"
    g2D_kernel = g2D_kernel_J1604

    # check that convolution happens for each spectral channel (correct dimension)
    data = img
    print(data.shape[0])
    
    conv_img = np.zeros(data.shape, np.float32) # new convolved image
    for c in range(data.shape[0]):
        conv_img[c,:,:] = convolve_fft(data[c, :, :], g2D_kernel)
    
    # convert units
    conv_img *= conversion_factor

    return conv_img



# ----------- Noising and Convolving -------------
# ------------------------------------------------
# source
filepath = "~/warped-disks/Lina_synth_cubes_mod/"
filename = "image_lines_incl30_02-twisted-wa5_mod.fits"

# write to
filepath_nonoise_convolved = "~/warped-disks/Elou_synth_cubes_mod_convolved/"
filepath_noisy_convolved = "~/warped-disks/Elou_synth_cubes_mod_noisy_convolved/"
filename_nonoise_convolved = 'image_lines_incl30_02-twisted-wa5_mod_no-noise_convolved.fits'
filename_noisy_convolved = 'image_lines_incl30_02-twisted-wa5_mod_noisy_convolved.fits' 

noise = True
convolve = True


pixsize_arcsec = pix_cm_to_arcsec(pixsize_cm, d_pc)


with fits.open(filepath+filename) as hdul:
    hdul.info()
    hdr = hdul[0].header
    img = hdul[0].data

    arcsec_min = -(img.shape[1]*pixsize_arcsec)/2 # convert x axis from pix to arcsec
    arcsec_max = (img.shape[1]*pixsize_arcsec)/2 # convert y axis from pix to arcsec

    # check image
    channel = 100
    plt.imshow(img[channel],  cmap = 'magma', extent = [arcsec_min, arcsec_max, arcsec_min, arcsec_max])
    plt.show()

    cube = img

    # add noise
    if noise == True:
        _, noisy_cube = add_noise(cube)
        # plt.imshow(noisy_cube[channel], cmap = 'magma', extent = [arcsec_min, arcsec_max, arcsec_min, arcsec_max])
        # plt.show()
        cube = noisy_cube
        #print(noisy_cube[0])

    # convolve with beam
    if convolve == True:
        conv_cube = convolve_img(cube)
        # plt.imshow(conv_cube[channel], cmap = 'magma', extent = [arcsec_min, arcsec_max, arcsec_min, arcsec_max])
        # plt.show()
        cube = conv_cube
        # print(conv_cube[0])

    # save
    hdu = fits.PrimaryHDU(data=cube, header=hdr)
    hdu.writeto(filepath_noisy_convolved+filename_noisy_convolved, overwrite = True)
    

# verify .fits
with fits.open(filepath_noisy_convolved+filename_noisy_convolved) as hdul:
    hdul.verify('fix')
    cube = hdul[0].data
    plt.imshow(cube[channel], cmap = 'magma', extent = [arcsec_min, arcsec_max, arcsec_min, arcsec_max])
    plt.show()
    #print(hdul[0].header)
