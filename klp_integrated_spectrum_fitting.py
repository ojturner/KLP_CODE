# This contains the functions necessary for fitting
# the integrated spectrum within object datacubes


# import the relevant modules
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyraf
import numpy.ma as ma
import pickle
from scipy.spatial import distance
from scipy import ndimage
from copy import copy
from lmfit import Model
from itertools import cycle as cycle
from lmfit.models import GaussianModel, PolynomialModel, ConstantModel
from scipy.optimize import minimize
from astropy.io import fits, ascii
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import MaxNLocator
from numpy import poly1d
from sys import stdout
from matplotlib import rc
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry
import astropy.units as u
from spectral_cube import SpectralCube

# add the class file to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk2/turner/disk2/turner/DATA/KLP/CODE')

import flatfield_cube as f_f
import psf_blurring as psf
import twod_gaussian as g2d
import rotate_pa as rt_pa
import aperture_growth as ap_growth
import make_table
import arctangent_1d as arc_mod
import oned_gaussian as one_d_g
import search_for_closest_sky as sky_search
import compute_smear_from_helen as dur_smear
import pa_calc
import integrated_spectrum_fitting as spec_fit
import mask_sky as mask_the_sky
import klp_spaxel_fitting as klp_spax_fit

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

# define speed of light
c = 2.99792458E5

def integrated_spec_extract(datacube,
                            central_x,
                            central_y,
                            redshift,
                            line,
                            aperture=6.0,
                            continuum_fit=True):

    """
    Def:
    Extract 1 arcsec diameter spectrum from the datacube
    in the given waveband and then fit it with the model 
    for that waveband after subtracting the continuum

    Input: datacube - stacked datacube corresponding to that object.
           central_x - vertical coordinate of the cube centre in the input file
           central_y - horizontal coordinate of the cube centre in the I.F

    Output: 
           Integrated spectrum full and fit to spectrum full
           will leave it to the klp_all_line_grids to decide
           how much of that spectrum to plot
    """
    # use the cubeclass to define a datacube
    cube = cubeOps(datacube)

    cube_data = cube.data
    wave_array = cube.wave_array

    # determine the central wavelength of the line
    # and hence the lower and upper limits to return

    if line == 'oiii':

        central_wl = 0.500824 * (1. + redshift)

    elif line == 'oiiiweak':

        central_wl = 0.4960295 * (1. + redshift)

    elif line == 'hb':

        central_wl = 0.486268 * (1. + redshift)

    elif line == 'oii':

        central_wl = 0.3728485 * (1. + redshift)

    elif line == 'ha':

        central_wl = 0.6564614 * (1. + redshift)

    elif line == 'nii':

        central_wl = 0.658527 * (1. + redshift)

    # find the index of the chosen emission line
    line_idx = np.argmin(np.abs(wave_array - central_wl))

    lower_index = line_idx - 75
    if lower_index < 0:
        lower_index = 0
    upper_index = line_idx + 75

    # define the x and y shapes
    x_shape = cube_data.shape[1]
    y_shape = cube_data.shape[2]

    filt = cube.filter

    # only subtract part that has been fit by polynomial
    if filt == 'K':
        xl = 100
        xu = 1750
        weight_name = datacube[:-5] + '_k_sky.fits'
        # also get the sky spectrum for weights
        weights = fits.open(weight_name)[0].data
    elif filt == 'H':
        xl = 100
        xu = 1750
        weight_name = datacube[:-5] + '_h_sky.fits'
        weights = fits.open(weight_name)[0].data
    elif filt == 'YJ':
        xl = 100
        xu = 1900
        weight_name = datacube[:-5] + '_yj_sky.fits'
        weights = fits.open(weight_name)[0].data

    # now extract the spectrum in the chosen aperture size

    spectral_cube = SpectralCube.read(datacube,hdu=1)

    # define the grid to use as a mask
    # in the K and H bands
    xx,yy = np.indices([x_shape,
                        y_shape], dtype='float')

    radius = ((yy-central_y)**2 + (xx-central_x)**2)**0.5

    mask = radius <= aperture

    # extract the data from the cube
    masked_cube = spectral_cube.with_mask(mask)

    # extract the spectrum from the unmasked region
    # must make sure this is a numpy array otherwise
    # nothing will work
    spectrum = np.array(masked_cube.sum(axis=(1,2)))

    # now subtract the continuum from this object spectrum
    # using the predefined helper function
    if continuum_fit:
        cont = continuum_subtract(spectrum,
                                  wave_array,
                                  filt,
                                  weights,
                                  redshift,
                                  xl,
                                  xu)
        spectrum = spectrum - cont

    # opportunity here to fit the integrated spectrum
    # but not going to do that tonight
    
    # testing to make sure the aperture is in the right place
    cont_image = np.nanmedian(cube_data[200:1800,:,:],axis=0)
#    fig, ax = plt.subplots(1,1,figsize=(8,8))
    vmin,vmax = np.nanpercentile(cont_image,[10.0, 90.0])
#    ax.imshow(cont_image,
#              vmin=vmin,
#              vmax=vmax)
#    ax.contour(yy,
#               xx,
#               mask,
#               1,
#               linewidths=3)
#    plt.show()
#    plt.close('all')
#    fig, ax = plt.subplots(1,1,figsize=(18,8))
#    ax.plot(wave_array[500:950],final_spec[500:950])
#    plt.show()
#    plt.close('all')

    return [spectrum[lower_index:upper_index],
            wave_array[lower_index:upper_index],
            cont_image, vmin, vmax, [yy,xx,mask]]

def continuum_subtract(spectrum,
                       wave_array,
                       filt,
                       weights,
                       redshift,
                       xl,
                       xu):

    """
    Def:
    Fit a polynomial to the noise spectrum, for the purpose of subtracting
    the thermal noise at the long wavelength end of the K-band.
    The noise spectrum is the sum of all spaxels not in the object spectrum,
    which are more than 5 pixels from the cube border.

    Input: 
            wave_array - the wavelength array of the spectrum
            final_spec - the summed spectrum

    Output:
            continuum subtracted final spectrum

    """

    # first mask the sky

    wave_array, sky_masked_spec = mask_the_sky.masking_sky(wave_array,
                                                           spectrum,
                                                           filt)

    # now mask the emission lines

    sky_and_emission_line_masked_spec = klp_spax_fit.mask_emission_lines(wave_array,
                                                                         sky_masked_spec,
                                                                         redshift,
                                                                         filt)

    bins = np.linspace(wave_array[xl], wave_array[xu], 400)
    delta = bins[1] - bins[0]
    idx = np.digitize(wave_array, bins)
    running_median = [np.nanmedian(sky_and_emission_line_masked_spec[idx==k]) for k in range(400)]

    # use lmfit to define the model and do the fitting
    poly_mod = PolynomialModel(7,missing='drop')
    pars = poly_mod.make_params()

    # for the masked array to work need to assign the parameters

    pars['c0'].set(value=1E-15)
    pars['c1'].set(value=1E-15)
    pars['c2'].set(value=1E-15)
    pars['c3'].set(value=1E-15)
    pars['c4'].set(value=1E-15)
    pars['c5'].set(value=1E-15)
    pars['c6'].set(value=1E-15)
    pars['c7'].set(value=1E-15)

#    fig, ax = plt.subplots(1,1,figsize=(18,8))
#    ax.plot(wave_array[xl:xu],sky_and_emission_line_masked_spec[xl:xu],drawstyle='steps-mid')
#    plt.show()
#    plt.close('all')
#    fig, ax = plt.subplots(1,1,figsize=(18,8))
#    ax.plot(bins-delta/2,running_median)
#    plt.show()
#    plt.close('all')

    out = poly_mod.fit(running_median,
                       pars,
                       x=bins-delta/2)
    poly_best = out.eval(x=wave_array)

#    fig, ax = plt.subplots(1,1,figsize=(18,8))
#    ax.plot(wave_array[100:1800],spectrum[100:1800],drawstyle='steps-mid')
#    ax.plot(wave_array[100:1800],poly_best[100:1800],drawstyle='steps-mid',color='red',lw=2)
#    plt.show()
#    plt.close('all')
    return poly_best

#construct_spec('/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/GS/K/P1/p1_comb/COMBINE_SCI_RECONSTRUCTED_GS3_19791.fits', 19.5, 19.5, 2.2254)