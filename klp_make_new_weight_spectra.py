# This class houses the methods which are relevant to manual additions to the
# ESO KMOS pipeline Mainly concentrating on two procedures - 1) pedestal
# readout column correction and 2) shifting and aligning sky and object
# images before data cube reconstruction


# import the relevant modules
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import pickle
import scipy.ndimage.filters as scifilt
from astropy.stats import median_absolute_deviation as astmad
from scipy.spatial import distance
from scipy import ndimage
from copy import copy
from lmfit import Model
from itertools import cycle as cycle
from lmfit.models import GaussianModel, PolynomialModel, ConstantModel
from scipy.optimize import minimize
from astropy.io import fits, ascii
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import poly1d
from sys import stdout
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry
import astropy.wcs.utils as autils
from astropy.wcs import WCS
from PIL import Image

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
import mask_sky as mask_the_sky
import klp_spaxel_fitting as spaxel_fit
import klp_integrated_spectrum_fitting as int_spec_fit

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

# create the weighting files for fitting gaussians to these spectra.
# how to do this in the case of integrated spectra which are summed
# over a certain region is much more puzzling, but seems doable
# for the fits to individual spaxels or individual regions

k_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Kband/KLP_K_NAMES_FINAL.txt')
k_name = k_table['Filename']
k_x_lower = k_table['mask_x_lower']
k_x_upper = k_table['mask_x_upper']
k_y_lower = k_table['mask_y_lower']
k_y_upper = k_table['mask_y_upper']

h_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Hband/KLP_H_NAMES_FINAL.txt')
h_name = h_table['Filename']
h_x_lower = h_table['mask_x_lower']
h_x_upper = h_table['mask_x_upper']
h_y_lower = h_table['mask_y_lower']
h_y_upper = h_table['mask_y_upper']

yj_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES_FINAL.txt')
yj_name = yj_table['Filename']
yj_x_lower = yj_table['mask_x_lower']
yj_x_upper = yj_table['mask_x_upper']
yj_y_lower = yj_table['mask_y_lower']
yj_y_upper = yj_table['mask_y_upper']

# loop round these, for each entry use the mask limits to
# extract background spaxels and hence define a weights spectrum

def make_new_weight_spectra_error_cube():

    for k_gal, k_x_l, k_x_u, k_y_l, k_y_u in zip(k_name,
                                                 k_x_lower,
                                                 k_x_upper,
                                                 k_y_lower,
                                                 k_y_upper):
        
        cube = cubeOps(k_gal)

        data = fits.open(k_gal)[2].data

        background_spaxel_list = []

        if k_x_l >= 6:
            for i in range(6,k_x_l + 1):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if k_x_u < data.shape[1]-7:
            for i in range(k_x_u - 1,data.shape[1]-7):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if k_y_l >= 6:
            for i in range(k_x_l,k_x_u):
                for j in range(6,k_y_l + 1):
                    background_spaxel_list.append(data[:,i,j])
        if k_y_u < data.shape[2]-7:
            for i in range(k_x_l,k_x_u):
                for j in range(k_y_u - 1,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])

        weights_spec = np.nanmedian(np.array(background_spaxel_list),axis=0)

        weights_spec_name = k_gal[:-5] + '_error_spectrum.fits'

        sky_hdu = fits.PrimaryHDU(weights_spec)

        sky_hdu.writeto(weights_spec_name,clobber=True)

    for h_gal, h_x_l, h_x_u, h_y_l, h_y_u in zip(h_name,
                                                 h_x_lower,
                                                 h_x_upper,
                                                 h_y_lower,
                                                 h_y_upper):
        
        cube = cubeOps(h_gal)

        data = fits.open(h_gal)[2].data

        background_spaxel_list = []

        if h_x_l >= 6:
            for i in range(6,h_x_l + 1):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if h_x_u < data.shape[1]-7:
            for i in range(h_x_u - 1,data.shape[1]-7):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if h_y_l >= 6:
            for i in range(h_x_l,h_x_u):
                for j in range(6,h_y_l + 1):
                    background_spaxel_list.append(data[:,i,j])
        if h_y_u < data.shape[2]-7:
            for i in range(h_x_l,h_x_u):
                for j in range(h_y_u - 1,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])

        weights_spec = np.nanmedian(np.array(background_spaxel_list),axis=0)

        weights_spec_name = h_gal[:-5] + '_error_spectrum.fits'

        sky_hdu = fits.PrimaryHDU(weights_spec)

        sky_hdu.writeto(weights_spec_name,clobber=True)

    for yj_gal, yj_x_l, yj_x_u, yj_y_l, yj_y_u in zip(yj_name,
                                                 yj_x_lower,
                                                 yj_x_upper,
                                                 yj_y_lower,
                                                 yj_y_upper):
        
        cube = cubeOps(yj_gal)

        data = fits.open(yj_gal)[2].data

        background_spaxel_list = []

        if yj_x_l >= 6:
            for i in range(6,yj_x_l + 1):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if yj_x_u < data.shape[1]-7:
            for i in range(yj_x_u - 1,data.shape[1]-7):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if yj_y_l >= 6:
            for i in range(yj_x_l,yj_x_u):
                for j in range(6,yj_y_l + 1):
                    background_spaxel_list.append(data[:,i,j])
        if yj_y_u < data.shape[2]-7:
            for i in range(yj_x_l,yj_x_u):
                for j in range(yj_y_u - 1,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])

        weights_spec = np.nanmedian(np.array(background_spaxel_list),axis=0)

        weights_spec_name = yj_gal[:-5] + '_error_spectrum.fits'

        sky_hdu = fits.PrimaryHDU(weights_spec)

        sky_hdu.writeto(weights_spec_name,clobber=True)

def make_new_weight_spectra_std():

    # pretty much doing the exact same as above
    # only using the datacube rather than the noisecube
    # and taking the standard deviation of the final values
    # rather than the median. Also going to get rid of the
    # lowest and highest values so that the standard deviation
    # won't be too badly affected by outliers

    for k_gal, k_x_l, k_x_u, k_y_l, k_y_u in zip(k_name,
                                                 k_x_lower,
                                                 k_x_upper,
                                                 k_y_lower,
                                                 k_y_upper):
        
        cube = cubeOps(k_gal)

        data = cube.data

        nan_list_data = []

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if np.isnan(data[i,j,k]):
                        data[i,j,k] = 0.0
                        nan_list_data.append([i,j,k])

        sigma_g = (0.2 / 0.1) / 2.355
        data = scifilt.gaussian_filter(data,
                                       sigma=[2.0,sigma_g,sigma_g])

        # reset the nan entries
        for entry in nan_list_data:
            data[entry[0],entry[1],entry[2]] = np.nan

        background_spaxel_list = []

        if k_x_l >= 6:
            for i in range(6,k_x_l + 1):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if k_x_u < data.shape[1]-7:
            for i in range(k_x_u - 1,data.shape[1]-7):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if k_y_l >= 6:
            for i in range(k_x_l,k_x_u):
                for j in range(6,k_y_l + 1):
                    background_spaxel_list.append(data[:,i,j])
        if k_y_u < data.shape[2]-7:
            for i in range(k_x_l,k_x_u):
                for j in range(k_y_u - 1,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])

        background_spaxel_list = np.array(background_spaxel_list)

        weights_spec = astmad(background_spaxel_list,ignore_nan=True,axis=0)

        weights_spec_name = k_gal[:-5] + '_error_spectrum022.fits'

        sky_hdu = fits.PrimaryHDU(weights_spec)

        sky_hdu.writeto(weights_spec_name,clobber=True)

    for h_gal, h_x_l, h_x_u, h_y_l, h_y_u in zip(h_name,
                                                 h_x_lower,
                                                 h_x_upper,
                                                 h_y_lower,
                                                 h_y_upper):
        
        cube = cubeOps(h_gal)

        data = cube.data

        nan_list_data = []

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if np.isnan(data[i,j,k]):
                        data[i,j,k] = 0.0
                        nan_list_data.append([i,j,k])

        sigma_g = (0.2 / 0.1) / 2.355
        data = scifilt.gaussian_filter(data,
                                       sigma=[2.0,sigma_g,sigma_g])

        # reset the nan entries
        for entry in nan_list_data:
            data[entry[0],entry[1],entry[2]] = np.nan

        background_spaxel_list = []

        if h_x_l >= 6:
            for i in range(6,h_x_l + 1):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if h_x_u < data.shape[1]-7:
            for i in range(h_x_u - 1,data.shape[1]-7):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if h_y_l >= 6:
            for i in range(h_x_l,h_x_u):
                for j in range(6,h_y_l + 1):
                    background_spaxel_list.append(data[:,i,j])
        if h_y_u < data.shape[2]-7:
            for i in range(h_x_l,h_x_u):
                for j in range(h_y_u - 1,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])

        background_spaxel_list = np.array(background_spaxel_list)

        weights_spec = astmad(background_spaxel_list,ignore_nan=True,axis=0)

        weights_spec_name = h_gal[:-5] + '_error_spectrum022.fits'

        sky_hdu = fits.PrimaryHDU(weights_spec)

        sky_hdu.writeto(weights_spec_name,clobber=True)

    for yj_gal, yj_x_l, yj_x_u, yj_y_l, yj_y_u in zip(yj_name,
                                                 yj_x_lower,
                                                 yj_x_upper,
                                                 yj_y_lower,
                                                 yj_y_upper):
        
        cube = cubeOps(yj_gal)

        data = cube.data

        nan_list_data = []

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if np.isnan(data[i,j,k]):
                        data[i,j,k] = 0.0
                        nan_list_data.append([i,j,k])

        sigma_g = (0.2 / 0.1) / 2.355
        data = scifilt.gaussian_filter(data,
                                       sigma=[2.0,sigma_g,sigma_g])

        # reset the nan entries
        for entry in nan_list_data:
            data[entry[0],entry[1],entry[2]] = np.nan

        background_spaxel_list = []

        if yj_x_l >= 6:
            for i in range(6,yj_x_l + 1):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if yj_x_u < data.shape[1]-7:
            for i in range(yj_x_u - 1,data.shape[1]-7):
                for j in range(6,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])
        if yj_y_l >= 6:
            for i in range(yj_x_l,yj_x_u):
                for j in range(6,yj_y_l + 1):
                    background_spaxel_list.append(data[:,i,j])
        if yj_y_u < data.shape[2]-7:
            for i in range(yj_x_l,yj_x_u):
                for j in range(yj_y_u - 1,data.shape[2]-7):
                    background_spaxel_list.append(data[:,i,j])

        background_spaxel_list = np.array(background_spaxel_list)

        weights_spec = astmad(background_spaxel_list,ignore_nan=True,axis=0)

        weights_spec_name = yj_gal[:-5] + '_error_spectrum022.fits'

        sky_hdu = fits.PrimaryHDU(weights_spec)

        sky_hdu.writeto(weights_spec_name,clobber=True)

make_new_weight_spectra_std()


