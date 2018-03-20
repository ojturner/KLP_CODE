# This contains the functions necessary for fitting
# the integrated spectrum within object datacubes


# import the relevant modules
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
                            continuum_fit=True,
                            prog='klp'):

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
        lower_index = 20
    upper_index = line_idx + 75

    # define the x and y shapes
    x_shape = cube_data.shape[1]
    y_shape = cube_data.shape[2]

    filt = cube.filter

    # only subtract part that has been fit by polynomial
    if filt == 'K':
        xl = 100
        xu = 1750
        weight_name = datacube[:-5] + '_error_spectrum.fits'
        # also get the sky spectrum for weights
        weights = fits.open(weight_name)[0].data
    elif filt == 'H':
        xl = 100
        xu = 1750
        weight_name = datacube[:-5] + '_error_spectrum.fits'
        weights = fits.open(weight_name)[0].data
    elif filt == 'YJ':
        xl = 100
        xu = 1900
        weight_name = datacube[:-5] + '_error_spectrum.fits'
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
                                  redshift,
                                  xl,
                                  xu,
                                  prog)
        spectrum = spectrum - cont

    # opportunity here to fit the integrated spectrum
    # but not going to do that tonight
    
    # testing to make sure the aperture is in the right place
    cont_image = np.nanmedian(cube_data[200:1800,:,:],axis=0)
#    fig, ax = plt.subplots(1,1,figsize=(8,8))
    vmin,vmax = np.nanpercentile(cont_image[7:x_shape-7,7:y_shape-7],[10.0, 90.0])
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
                       redshift,
                       xl,
                       xu,
                       prog):

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
                                                                         filt,
                                                                         prog)

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

def line_fit(wave_array,
             spec,
             weights,
             redshift,
             filt,
             name,
             plot_save_name,
             masked=False,
             prog='klp'):

    """
    Def:
    Fitting the KLP emission lines in an integrated sense.
    Construct a model with three gaussian components and fit to the integrated
    spectrum. Return the gaussian fit parameters. Lots of freedom for how the
    gaussian fit should be done, how the centres should be constrained with
    respect to one another and what happens when one of the emission lines is
    not there.
    """
    print 'FITTING: %s' % name
    # global tolerance for masking the lines in noise computation
    tol_global = 0.0025
    # line wavelengths
    h_alpha_rest = 0.6564614
    h_alpha_shifted = (1 + redshift) * h_alpha_rest
    h_alpha_shifted_min = h_alpha_shifted - tol_global
    h_alpha_shifted_max = h_alpha_shifted + tol_global
    nii_rest = 0.658523
    nii_shifted = (1 + redshift) * nii_rest
    nii_shifted_min = nii_shifted - tol_global
    nii_shifted_max = nii_shifted + tol_global
    h_beta_rest = 0.4862721
    h_beta_shifted = (1 + redshift) * h_beta_rest
    h_beta_shifted_min = h_beta_shifted - tol_global
    h_beta_shifted_max = h_beta_shifted + tol_global
    oiii_4960_rest = 0.4960295
    oiii_4960_shifted = (1 + redshift) * oiii_4960_rest
    oiii_4960_shifted_min = oiii_4960_shifted - tol_global
    oiii_4960_shifted_max = oiii_4960_shifted + tol_global
    oiii_5008_rest = 0.5008239
    oiii_5008_shifted = (1 + redshift) * oiii_5008_rest
    oiii_5008_shifted_min = oiii_5008_shifted - tol_global
    oiii_5008_shifted_max = oiii_5008_shifted + tol_global
    oii_3727_rest = 0.3727092
    oii_3727_shifted = (1 + redshift) * oii_3727_rest
    oii_3727_shifted_min = oii_3727_shifted - tol_global
    oii_3727_shifted_max = oii_3727_shifted + tol_global
    oii_3729_rest = 0.3729875
    oii_3729_shifted = (1 + redshift) * oii_3729_rest
    oii_3729_shifted_min = oii_3729_shifted - tol_global
    oii_3729_shifted_max = oii_3729_shifted + tol_global

    # get the masked arrays from the mask_sky method
    # only if the masked thing is set to true
    if masked:
        wave_array, spec = mask_sky(wave_array,
                                    spec,
                                    filt)

    # choose the appropriate sky dictionary for the filter
    if filt == 'K':
        print 'K-band fitting'
        sky_dict = mask_the_sky.ret_k_sky()
        tol = 0.00028
        comp_mod, pars, min_index, max_index = k_band_mod(tol,
                                                          redshift,
                                                          wave_array,
                                                          prog)

    elif filt == 'H':
        print 'H-band fitting'
        sky_dict = mask_the_sky.ret_h_sky()
        tol = 0.00028
        comp_mod, pars, min_index, max_index = h_band_mod(tol,
                                                          redshift,
                                                          wave_array,
                                                          prog)

    elif filt == 'YJ':
        print 'YJ-band fitting'
        sky_dict = mask_the_sky.ret_yj_sky()
        tol = 0.00028
        comp_mod, pars, min_index, max_index = yj_band_mod(tol,
                                                           redshift,
                                                           wave_array)

    # also return a total masked spectrum which blocks out
    # emission lines

    spec_total = klp_spax_fit.mask_emission_lines(wave_array,
                                                  spec,
                                                  redshift,
                                                  filt,
                                                  prog)

    # noise calculation
    # as part of this process also need to know the signal to noise level
    # of the observations. Fit a histogram of the observations with a
    # gaussian model centred on 0 and return the standard deviation
    # of that as the noise

    # need to do this considering both fit-filt and filt again

    masked_spectrum = spec_total[100:1700].compressed()
    
    masked_median = np.nanmedian(masked_spectrum)
    masked_spectrum = masked_spectrum / masked_median
    # take only the 5th to 95th percentile
    masked_spectrum = np.sort(masked_spectrum)
    masked_spectrum = masked_spectrum[int(len(masked_spectrum)/20.0):int(0.95*len(masked_spectrum))]
    bins, centres = np.histogram(masked_spectrum, bins=20)
    noise_mod = GaussianModel()
    noise_pars = noise_mod.guess(bins, x=centres[:-1])
    noise_result = noise_mod.fit(bins, params=noise_pars, x=centres[:-1])

    #print noise_result.fit_report()
    #print bins, centres
    noise_best_fit = noise_result.eval(x=centres[:-1])

    fig, ax = plt.subplots(1,1,figsize=(8,10))
    ax.scatter(centres[:-1],bins)
    ax.plot(centres[:-1],noise_best_fit)
    #plt.show()
    plt.close('all')

    noise = abs(noise_result.best_values['sigma']*masked_median)

    # also want to subtract the centre of the noise array from the 
    # spectrum to ensure signal to noise is computed correctly.

    spec = spec - noise_result.best_values['center']*masked_median

    # normalise the weights by the median value
    # weights_norm = weights/np.median(weights)

    # get a median spectrum to normalise the data and the weights
    med_for_norm = abs(np.nanmedian(spec[min_index:max_index]))
    weights_norm = weights/med_for_norm

    result =  comp_mod.fit(spec[min_index:max_index]/med_for_norm,
                           params=pars,
                           x=wave_array[min_index:max_index],
                           weights=(1.0/weights_norm[min_index:max_index])**2)

#    print result.fit_report()
#    res_plot = result.plot()
#    plt.show(res_plot)
#    plt.close(res_plot)

    # another point at which we must discriminate between KLP and KDS
    if prog == 'klp':
        if  filt == 'K':
            ha_rescaled = (result.best_values['ha_amplitude']*med_for_norm) / (result.best_values['ha_sigma'] * np.sqrt(2 * np.pi))
            ha_sn = 1.0*ha_rescaled / noise
            try:
                ha_error = np.sqrt(result.covar[2,2])*med_for_norm
            except TypeError:
                ha_error = -99
            ha_sigma = (result.best_values['ha_sigma'] / result.best_values['ha_center']) * c
            try:
                ha_sigma_error = ( np.sqrt(result.covar[1,1]) / result.best_values['ha_center']) * c
            except:
                ha_sigma_error = -99
            print 'Ha SN %s' % (ha_sn)
            nii_rescaled = (result.best_values['nii_amplitude']*med_for_norm) / (result.best_values['nii_sigma'] * np.sqrt(2 * np.pi))
            nii_sn = 1.0*nii_rescaled / noise
            try:
                nii_error = np.sqrt(result.covar[0,0])*med_for_norm
            except TypeError:
                nii_error = -99
            nii_sigma = (result.best_values['nii_sigma'] / result.best_values['nii_center']) * c
            try:
                nii_sigma_error = ( np.sqrt(result.covar[3,3]) / result.best_values['nii_center']) * c
            except TypeError:
                nii_sigma_error = -99
            print 'NII SN %s' % (nii_sn)
            output_array = np.array([[result.best_values['ha_amplitude']*med_for_norm,ha_error,ha_sn,ha_sigma,ha_sigma_error],
                            [result.best_values['nii_amplitude']*med_for_norm,nii_error,nii_sn,nii_sigma,nii_sigma_error]])
            ylim_lower = -0.2*(result.best_values['ha_amplitude']*med_for_norm/result.best_values['ha_sigma'])
            ylim_upper = (result.best_values['ha_amplitude']*med_for_norm/result.best_values['ha_sigma']) + 0.3*(result.best_values['ha_amplitude']*med_for_norm/result.best_values['ha_sigma'])
            constant = 0
        elif filt == 'YJ':
            oii_3727_rescaled = (result.best_values['oiil_amplitude']*med_for_norm) / (result.best_values['oiil_sigma'] * np.sqrt(2 * np.pi))
            oii_3727_sn = (1.0*oii_3727_rescaled / noise)
            try:
                oii_3727_error = np.sqrt(result.covar[2,2])*med_for_norm
            except TypeError:
                oii_3727_error = -99
            oii_3727_sigma = (result.best_values['oiil_sigma'] / result.best_values['oiil_center']) * c    
            try:
                oii_3727_sigma_error = (np.sqrt(result.covar[0,0]) / result.best_values['oiil_center']) * c
            except TypeError:
                oii_3727_sigma_error = -99
            print 'OII3727 SN %s' % (oii_3727_sn)
            oii_3729_rescaled = (result.best_values['oiih_amplitude']*med_for_norm) / (result.best_values['oiih_sigma'] * np.sqrt(2 * np.pi))
            oii_3729_sn = (1.0*oii_3729_rescaled / noise)
            try:
                oii_3729_error = np.sqrt(result.covar[3,3])*med_for_norm
            except TypeError:
                oii_3729_error = -99
            oii_3729_sigma = (result.best_values['oiih_sigma'] / result.best_values['oiih_center']) * c    
            try:
                oii_3729_sigma_error = (np.sqrt(result.covar[0,0]) / result.best_values['oiih_center']) * c
            except TypeError:
                oii_3729_sigma_error = -99
            print 'OII3729 SN %s' % (oii_3729_sn)
            output_array = np.array([[result.best_values['oiil_amplitude']*med_for_norm,oii_3727_error,oii_3727_sn,oii_3727_sigma,oii_3727_sigma_error],
                                     [result.best_values['oiih_amplitude']*med_for_norm,oii_3729_error,oii_3729_sn,oii_3729_sigma,oii_3729_sigma_error]])
            ylim_lower = -0.2*(result.best_values['oiil_amplitude']*med_for_norm/result.best_values['oiil_sigma'])
            ylim_upper = (result.best_values['oiil_amplitude']*med_for_norm/result.best_values['oiil_sigma'])+0.3*(result.best_values['oiil_amplitude']*med_for_norm/result.best_values['oiil_sigma'])
            constant = result.best_values['c']
        elif filt == 'H':
            hb_rescaled = (result.best_values['hb_amplitude']*med_for_norm) / (result.best_values['hb_sigma'] * np.sqrt(2 * np.pi))
            hb_sn = 1.0*hb_rescaled / noise
            try:
                hb_error = np.sqrt(result.covar[4,4])*med_for_norm
            except TypeError:
                hb_error = -99
            hb_sigma = (result.best_values['hb_sigma'] / result.best_values['hb_center']) * c
            try:
                hb_sigma_error = ( np.sqrt(result.covar[0,0]) / result.best_values['hb_center']) * c
            except:
                hb_sigma_error = -99
            print 'Hb SN %s' % (hb_sn)
            oiii_4960_rescaled = (result.best_values['oiii4_amplitude']*med_for_norm) / (result.best_values['oiii4_sigma'] * np.sqrt(2 * np.pi))
            oiii_4960_sn = 1.0*oiii_4960_rescaled / noise
            try:
                oiii_4960_error = np.sqrt(result.covar[1,1])*med_for_norm
            except TypeError:
                oiii_4960_error = -99
            oiii_4960_sigma = (result.best_values['oiii4_sigma'] / result.best_values['oiii4_center']) * c
            try:
                oiii_4960_sigma_error = ( np.sqrt(result.covar[5,5]) / result.best_values['oiii4_center']) * c
            except TypeError:
                oiii_4960_sigma_error = -99
            print 'OIII4960 SN %s' % (oiii_4960_sn)
            oiii_5007_rescaled = (result.best_values['oiii5_amplitude']*med_for_norm) / (result.best_values['oiii5_sigma'] * np.sqrt(2 * np.pi))
            oiii_5007_sn = 1.0*oiii_5007_rescaled / noise
            try:
                oiii_5007_error = np.sqrt(result.covar[1,1])*med_for_norm
            except TypeError:
                oiii_5007_error = -99
            oiii_5007_sigma = (result.best_values['oiii5_sigma'] / result.best_values['oiii5_center']) * c
            try:
                oiii_5007_sigma_error = (np.sqrt(result.covar[5,5]) / result.best_values['oiii5_center']) * c
            except TypeError:
                oiii_5007_sigma_error = -99
            print 'OIII5007 SN %s' % (oiii_5007_sn)
            print 'NOISE %s' % noise
            print 'SIGNAL %s' % oiii_5007_rescaled
            output_array = np.array([[result.best_values['hb_amplitude']*med_for_norm,hb_error,hb_sn,hb_sigma,hb_sigma_error],
                            [result.best_values['oiii5_amplitude']*med_for_norm,oiii_5007_error,oiii_5007_sn,oiii_5007_sigma,oiii_5007_sigma_error],
                            [result.best_values['oiii4_amplitude']*med_for_norm,oiii_4960_error,oiii_4960_sn,oiii_4960_sigma,oiii_4960_sigma_error]])
            ylim_lower = -0.1*result.best_values['oiii5_amplitude']*med_for_norm/result.best_values['oiii5_sigma']
            ylim_upper = (result.best_values['oiii5_amplitude']*med_for_norm/result.best_values['oiii5_sigma']) + 0.3*(result.best_values['oiii5_amplitude']*med_for_norm/result.best_values['oiii5_sigma'])
            constant = result.best_values['c']

    else:
        if  filt == 'K':
            hb_rescaled = (result.best_values['hb_amplitude']*med_for_norm) / (result.best_values['hb_sigma'] * np.sqrt(2 * np.pi))
            hb_sn = 1.0*hb_rescaled / noise
            try:
                hb_error = np.sqrt(result.covar[4,4])*med_for_norm
            except TypeError:
                hb_error = -99
            hb_sigma = (result.best_values['hb_sigma'] / result.best_values['hb_center']) * c
            try:
                hb_sigma_error = ( np.sqrt(result.covar[0,0]) / result.best_values['hb_center']) * c
            except:
                hb_sigma_error = -99
            print 'Hb SN %s' % (hb_sn)
            oiii_4960_rescaled = (result.best_values['oiii4_amplitude']*med_for_norm) / (result.best_values['oiii4_sigma'] * np.sqrt(2 * np.pi))
            oiii_4960_sn = 1.0*oiii_4960_rescaled / noise
            try:
                oiii_4960_error = np.sqrt(result.covar[1,1])*med_for_norm
            except TypeError:
                oiii_4960_error = -99
            oiii_4960_sigma = (result.best_values['oiii4_sigma'] / result.best_values['oiii4_center']) * c
            try:
                oiii_4960_sigma_error = ( np.sqrt(result.covar[5,5]) / result.best_values['oiii4_center']) * c
            except TypeError:
                oiii_4960_sigma_error = -99
            print 'OIII4960 SN %s' % (oiii_4960_sn)
            oiii_5007_rescaled = (result.best_values['oiii5_amplitude']*med_for_norm) / (result.best_values['oiii5_sigma'] * np.sqrt(2 * np.pi))
            oiii_5007_sn = 1.0*oiii_5007_rescaled / noise
            try:
                oiii_5007_error = np.sqrt(result.covar[1,1])*med_for_norm
            except TypeError:
                oiii_5007_error = -99
            oiii_5007_sigma = (result.best_values['oiii5_sigma'] / result.best_values['oiii5_center']) * c
            try:
                oiii_5007_sigma_error = (np.sqrt(result.covar[5,5]) / result.best_values['oiii5_center']) * c
            except TypeError:
                oiii_5007_sigma_error = -99
            print 'OIII5007 SN %s' % (oiii_5007_sn)
            print 'NOISE %s' % noise
            print 'SIGNAL %s' % oiii_5007_rescaled
            output_array = np.array([[result.best_values['hb_amplitude']*med_for_norm,hb_error,hb_sn,hb_sigma,hb_sigma_error],
                            [result.best_values['oiii5_amplitude']*med_for_norm,oiii_5007_error,oiii_5007_sn,oiii_5007_sigma,oiii_5007_sigma_error],
                            [result.best_values['oiii4_amplitude']*med_for_norm,oiii_4960_error,oiii_4960_sn,oiii_4960_sigma,oiii_4960_sigma_error]])
            ylim_lower = -0.1*result.best_values['oiii5_amplitude']*med_for_norm/result.best_values['oiii5_sigma']
            ylim_upper = (result.best_values['oiii5_amplitude']*med_for_norm/result.best_values['oiii5_sigma']) + 0.3*(result.best_values['oiii5_amplitude']*med_for_norm/result.best_values['oiii5_sigma'])
            constant = result.best_values['c']
        elif filt == 'H':
            oii_3727_rescaled = (result.best_values['oiil_amplitude']*med_for_norm) / (result.best_values['oiil_sigma'] * np.sqrt(2 * np.pi))
            oii_3727_sn = (1.0*oii_3727_rescaled / noise)
            try:
                oii_3727_error = np.sqrt(result.covar[2,2])*med_for_norm
            except TypeError:
                oii_3727_error = -99
            oii_3727_sigma = (result.best_values['oiil_sigma'] / result.best_values['oiil_center']) * c    
            try:
                oii_3727_sigma_error = (np.sqrt(result.covar[0,0]) / result.best_values['oiil_center']) * c
            except TypeError:
                oii_3727_sigma_error = -99
            print 'OII3727 SN %s' % (oii_3727_sn)
            oii_3729_rescaled = (result.best_values['oiih_amplitude']*med_for_norm) / (result.best_values['oiih_sigma'] * np.sqrt(2 * np.pi))
            oii_3729_sn = (1.0*oii_3729_rescaled / noise)
            try:
                oii_3729_error = np.sqrt(result.covar[3,3])*med_for_norm
            except TypeError:
                oii_3729_error = -99
            oii_3729_sigma = (result.best_values['oiih_sigma'] / result.best_values['oiih_center']) * c    
            try:
                oii_3729_sigma_error = (np.sqrt(result.covar[0,0]) / result.best_values['oiih_center']) * c
            except TypeError:
                oii_3729_sigma_error = -99
            print 'OII3729 SN %s' % (oii_3729_sn)
            output_array = np.array([[result.best_values['oiil_amplitude']*med_for_norm,oii_3727_error,oii_3727_sn,oii_3727_sigma,oii_3727_sigma_error],
                                     [result.best_values['oiih_amplitude']*med_for_norm,oii_3729_error,oii_3729_sn,oii_3729_sigma,oii_3729_sigma_error]])
            ylim_lower = -0.2*(result.best_values['oiil_amplitude']*med_for_norm/result.best_values['oiil_sigma'])
            ylim_upper = (result.best_values['oiil_amplitude']*med_for_norm/result.best_values['oiil_sigma'])+0.3*(result.best_values['oiil_amplitude']*med_for_norm/result.best_values['oiil_sigma'])
            constant = result.best_values['c']

    best_fit = result.eval(x=wave_array)*med_for_norm
    ylim_upper = np.nanmax(best_fit[min_index:max_index])+0.2*np.nanmax(best_fit[min_index:max_index])

    line_fit_fig, line_fit_ax = plt.subplots(2,1,figsize=(18,8), sharex=True, gridspec_kw = {'height_ratios':[4,1]})

    lines = {'linestyle': '-'}
    plt.rc('lines', **lines)

    line_fit_ax[0].plot(wave_array[100:1900],spec[100:1900],drawstyle='steps-mid')
    line_fit_ax[0].plot(wave_array[100:1900],best_fit[100:1900],color='red',drawstyle='steps-mid',lw=2)
    line_fit_ax[1].plot(wave_array[100:1900],weights[100:1900],color='green',drawstyle='steps-mid',lw=2)
    line_fit_ax[0].plot(wave_array[100:1900],best_fit[100:1900],color='red')
    line_fit_ax[0].fill_between(wave_array[100:1900],
                                np.repeat(constant,len(wave_array[100:1900])),
                                best_fit[100:1900],
                                facecolor='red',
                                edgecolor='red',
                                alpha=0.5)

    #ax[0].set_ylim(-0.2,1.0)
    line_fit_ax[0].set_xlim(wave_array[min_index],wave_array[max_index])
    line_fit_ax[0].set_ylim(ylim_lower,ylim_upper)
    line_fit_ax[1].set_ylim(np.nanmin(weights[min_index:max_index]),np.nanmax(weights[min_index:max_index]))

    for ranges in sky_dict.values():
        line_fit_ax[0].axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')

    fig.subplots_adjust(hspace=.0)
    yticks = line_fit_ax[0].yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)

    # tick parameters 
    line_fit_ax[0].tick_params(axis='both',
                   which='major',
                   labelsize=26,
                   length=12,
                   width=4)

    line_fit_ax[0].tick_params(axis='both',
                   which='minor',
                   labelsize=26,
                   length=6,
                   width=4)

    line_fit_ax[0].minorticks_on()
    line_fit_ax[0].yaxis.get_offset_text().set_fontsize(26)

    [i.set_linewidth(4.0) for i in line_fit_ax[0].spines.itervalues()]
    # tick parameters 
    line_fit_ax[1].tick_params(axis='both',
                   which='major',
                   labelsize=18,
                   length=12,
                   width=4)

    line_fit_ax[1].tick_params(axis='both',
                   which='minor',
                   labelsize=18,
                   length=6,
                   width=4)

    [i.set_linewidth(4.0) for i in line_fit_ax[1].spines.itervalues()]
#    ax[0].yaxis.set_major_locator(MaxNLocator(prune='lower'))
#    ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))
        
    plt.show(line_fit_fig)
    line_fit_fig.savefig(plot_save_name)
    plt.close(line_fit_fig)

    return output_array

def k_band_mod(tol,
               redshift,
               wave_array,
               prog):

    """
    Def:
    Returns the k-band lmfit model and model parameters.
    This includes the OIII lines and Hbeta (appropriate for KDS redshift range)
    """

    if prog == 'klp':
        # line wavelengths
        h_alpha_rest = 0.6564614
        h_alpha_shifted = (1 + redshift) * h_alpha_rest
        h_alpha_shifted_min = h_alpha_shifted - tol
        h_alpha_shifted_max = h_alpha_shifted + tol
        nii_rest = 0.658523
        nii_shifted = (1 + redshift) * nii_rest
        nii_shifted_min = nii_shifted - tol
        nii_shifted_max = nii_shifted + tol

        # find the fitting range
        fitting_range_limit = 0.01
        min_index = np.argmin(abs(wave_array - (h_alpha_shifted - fitting_range_limit)))
        max_index = np.argmin(abs(wave_array - (nii_shifted + fitting_range_limit)))

        # construct a composite gaussian model with prefix parameter names
        comp_mod = GaussianModel(missing='drop',
                                 prefix='ha_') + \
                   GaussianModel(missing='drop',
                                 prefix='nii_') + \
                   ConstantModel(missing='drop')

        # set the wavelength value with min and max range
        # and initialise the other parameters

        comp_mod.set_param_hint('ha_center',
                                value=h_alpha_shifted,
                                min=h_alpha_shifted_min,
                                max=h_alpha_shifted_max)
        comp_mod.set_param_hint('ha_amplitude',
                                value=0.1,
                                min=0)
        comp_mod.set_param_hint('ha_sigma',
                                value=0.001,
                                min=0)
        comp_mod.set_param_hint('nii_center',
                                value=nii_shifted,
                                min=nii_shifted_min,
                                max=nii_shifted_max)
        comp_mod.set_param_hint('nii_amplitude',
                                value=0.01,
                                min=0)
        comp_mod.set_param_hint('nii_sigma',
                                value=0.001)

        comp_mod.set_param_hint('c',
                                value=0.0,
                                vary=False)

        pars = comp_mod.make_params()
    else:
        # line wavelengths
        h_beta_rest = 0.4862721
        h_beta_shifted = (1 + redshift) * h_beta_rest
        h_beta_shifted_min = h_beta_shifted - tol
        h_beta_shifted_max = h_beta_shifted + tol
        oiii_4960_rest = 0.4960295
        oiii_4960_shifted = (1 + redshift) * oiii_4960_rest
        oiii_4960_shifted_min = oiii_4960_shifted - tol
        oiii_4960_shifted_max = oiii_4960_shifted + tol
        oiii_5008_rest = 0.5008239
        oiii_5008_shifted = (1 + redshift) * oiii_5008_rest
        oiii_5008_shifted_min = oiii_5008_shifted - tol
        oiii_5008_shifted_max = oiii_5008_shifted + tol

        # wavelength separation of the two doubly ionised oxygen lines
        delta_oiii = oiii_5008_rest - oiii_4960_rest

        # find the fitting range
        fitting_range_limit = 0.01
        min_index = np.argmin(abs(wave_array - (h_beta_shifted - fitting_range_limit)))
        max_index = np.argmin(abs(wave_array - (oiii_5008_shifted + fitting_range_limit)))

        # construct a composite gaussian model with prefix parameter names
        comp_mod = GaussianModel(missing='drop',
                                 prefix='hb_') + \
                   GaussianModel(missing='drop',
                                 prefix='oiii4_') + \
                   GaussianModel(missing='drop',
                                 prefix='oiii5_') + \
                   ConstantModel(missing='drop')
        # set the wavelength value with min and max range
        # and initialise the other parameters

        comp_mod.set_param_hint('hb_center',
                                value=h_beta_shifted,
                                min=h_beta_shifted_min,
                                max=h_beta_shifted_max)
        comp_mod.set_param_hint('hb_amplitude',
                                value=0.01,
                                min=0)
        comp_mod.set_param_hint('oiii4_center',
                                value=oiii_4960_shifted,
                                expr='oiii5_center - ((%.6f)*%.6f)' % (1.+redshift,delta_oiii))
        comp_mod.set_param_hint('oiii4_amplitude',
                                value=0.3,
                                expr='0.3448*oiii5_amplitude')
        comp_mod.set_param_hint('oiii5_center',
                                value=oiii_5008_shifted,
                                min=oiii_5008_shifted_min,
                                max=oiii_5008_shifted_max)
        comp_mod.set_param_hint('oiii5_amplitude',
                                value=0.9,
                                min=0)
        comp_mod.set_param_hint('oiii5_sigma',
                                value=0.001,
                                min=0.0004,
                                max=0.002)
        comp_mod.set_param_hint('oiii4_sigma',
                                value=0.001,
                                expr='1.*oiii5_sigma')
        comp_mod.set_param_hint('hb_sigma',
                                value=0.001,
                                min=0.0004,
                                max=0.0016)

        comp_mod.set_param_hint('c',
                                value=0.0,
                                vary=False)

        pars = comp_mod.make_params()

    return [comp_mod,pars,min_index,max_index]


def h_band_mod(tol,
               redshift,
               wave_array,
               prog):

    """
    Def:
    Returns the k-band lmfit model and model parameters.
    This includes the OIII lines and Hbeta (appropriate for KDS redshift range)
    """

    if prog == 'klp':
        # line wavelengths
        h_beta_rest = 0.4862721
        h_beta_shifted = (1 + redshift) * h_beta_rest
        h_beta_shifted_min = h_beta_shifted - tol
        h_beta_shifted_max = h_beta_shifted + tol
        oiii_4960_rest = 0.4960295
        oiii_4960_shifted = (1 + redshift) * oiii_4960_rest
        oiii_4960_shifted_min = oiii_4960_shifted - tol
        oiii_4960_shifted_max = oiii_4960_shifted + tol
        oiii_5008_rest = 0.5008239
        oiii_5008_shifted = (1 + redshift) * oiii_5008_rest
        oiii_5008_shifted_min = oiii_5008_shifted - tol
        oiii_5008_shifted_max = oiii_5008_shifted + tol

        # wavelength separation of the two doubly ionised oxygen lines
        delta_oiii = oiii_5008_rest - oiii_4960_rest

        # find the fitting range
        fitting_range_limit = 0.01
        min_index = np.argmin(abs(wave_array - (h_beta_shifted - fitting_range_limit)))
        max_index = np.argmin(abs(wave_array - (oiii_5008_shifted + fitting_range_limit)))

        # construct a composite gaussian model with prefix parameter names
        comp_mod = GaussianModel(missing='drop',
                                 prefix='hb_') + \
                   GaussianModel(missing='drop',
                                 prefix='oiii4_') + \
                   GaussianModel(missing='drop',
                                 prefix='oiii5_') + \
                   ConstantModel(missing='drop')
        # set the wavelength value with min and max range
        # and initialise the other parameters

        comp_mod.set_param_hint('hb_center',
                                value=h_beta_shifted,
                                min=h_beta_shifted_min,
                                max=h_beta_shifted_max)
        comp_mod.set_param_hint('hb_amplitude',
                                value=0.01,
                                min=0)
        comp_mod.set_param_hint('oiii4_center',
                                value=oiii_4960_shifted,
                                expr='oiii5_center - ((%.6f)*%.6f)' % (1.+redshift,delta_oiii))
        comp_mod.set_param_hint('oiii4_amplitude',
                                value=0.3,
                                expr='0.3448*oiii5_amplitude')
        comp_mod.set_param_hint('oiii5_center',
                                value=oiii_5008_shifted,
                                min=oiii_5008_shifted_min,
                                max=oiii_5008_shifted_max)
        comp_mod.set_param_hint('oiii5_amplitude',
                                value=0.9,
                                min=0)
        comp_mod.set_param_hint('oiii5_sigma',
                                value=0.001,
                                min=0.0004,
                                max=0.002)
        comp_mod.set_param_hint('oiii4_sigma',
                                value=0.001,
                                expr='1.*oiii5_sigma')
        comp_mod.set_param_hint('hb_sigma',
                                value=0.001,
                                min=0.0004,
                                max=0.0016)

        comp_mod.set_param_hint('c',
                                value=0.0,
                                vary=False)

        pars = comp_mod.make_params()
    else:
        # define the line wavelengths
        oii_3727_rest = 0.3727092
        oii_3727_shifted = (1 + redshift) * oii_3727_rest
        oii_3727_shifted_min = oii_3727_shifted - tol
        oii_3727_shifted_max = oii_3727_shifted + tol

        # note we'll use an expression to define the position of this
        # rather than defining minimum and max and shifted
        oii_3729_rest = 0.3729875
        oii_3729_shifted = (1 + redshift) * oii_3729_rest
        oii_3729_shifted_min = oii_3729_shifted - tol
        oii_3729_shifted_max = oii_3729_shifted + tol

        # separation between the two
        delta_oii = oii_3729_rest - oii_3727_rest

        # find the fitting range
        fitting_range_limit = 0.01
        min_index = np.argmin(abs(wave_array - (oii_3727_shifted - fitting_range_limit)))
        max_index = np.argmin(abs(wave_array - (oii_3729_shifted + fitting_range_limit)))

        # construct a composite gaussian model with prefix parameter names
        comp_mod = GaussianModel(missing='drop',
                                 prefix='oiil_') + \
                   GaussianModel(missing='drop',
                                 prefix='oiih_') + \
                   ConstantModel(missing='drop')
        # set the wavelength value with min and max range
        # and initialise the other parameters

        comp_mod.set_param_hint('oiil_center',
                                value=oii_3727_shifted,
                                min=oii_3727_shifted_min,
                                max=oii_3727_shifted_max)
        comp_mod.set_param_hint('oiil_amplitude',
                                value=0.01,
                                min=0)
        comp_mod.set_param_hint('oiil_sigma',
                                value=0.0005,
                                expr='1.*oiih_sigma')
        comp_mod.set_param_hint('oiih_center',
                                value=oii_3729_shifted,
                                expr='((%.6f)*%.6f) + oiil_center' % (1.+redshift,delta_oii))
        comp_mod.set_param_hint('oiih_sigma',
                                value=0.0005,
                                min=0.000,
                                max=0.003)
        comp_mod.set_param_hint('oiih_amplitude',
                                value=0.01,
                                min=0)


        comp_mod.set_param_hint('c',
                                value=0.0,
                                vary=False)

        pars = comp_mod.make_params()


    return [comp_mod,pars,min_index,max_index]

def yj_band_mod(tol,
                redshift,
                wave_array):

    """
    Def:
    Returns the k-band lmfit model and model parameters.
    This includes the OIII lines and Hbeta (appropriate for KDS redshift range)
    """
    # define the line wavelengths
    oii_3727_rest = 0.3727092
    oii_3727_shifted = (1 + redshift) * oii_3727_rest
    oii_3727_shifted_min = oii_3727_shifted - tol
    oii_3727_shifted_max = oii_3727_shifted + tol

    # note we'll use an expression to define the position of this
    # rather than defining minimum and max and shifted
    oii_3729_rest = 0.3729875
    oii_3729_shifted = (1 + redshift) * oii_3729_rest
    oii_3729_shifted_min = oii_3729_shifted - tol
    oii_3729_shifted_max = oii_3729_shifted + tol

    # separation between the two
    delta_oii = oii_3729_rest - oii_3727_rest

    # find the fitting range
    fitting_range_limit = 0.01
    min_index = np.argmin(abs(wave_array - (oii_3727_shifted - fitting_range_limit)))
    max_index = np.argmin(abs(wave_array - (oii_3729_shifted + fitting_range_limit)))

    # construct a composite gaussian model with prefix parameter names
    comp_mod = GaussianModel(missing='drop',
                             prefix='oiil_') + \
               GaussianModel(missing='drop',
                             prefix='oiih_') + \
               ConstantModel(missing='drop')
    # set the wavelength value with min and max range
    # and initialise the other parameters

    comp_mod.set_param_hint('oiil_center',
                            value=oii_3727_shifted,
                            min=oii_3727_shifted_min,
                            max=oii_3727_shifted_max)
    comp_mod.set_param_hint('oiil_amplitude',
                            value=0.01,
                            min=0)
    comp_mod.set_param_hint('oiil_sigma',
                            value=0.0005,
                            expr='1.*oiih_sigma')
    comp_mod.set_param_hint('oiih_center',
                            value=oii_3729_shifted,
                            expr='((%.6f)*%.6f) + oiil_center' % (1.+redshift,delta_oii))
    comp_mod.set_param_hint('oiih_sigma',
                            value=0.0005,
                            min=0.000,
                            max=0.003)
    comp_mod.set_param_hint('oiih_amplitude',
                            value=0.01,
                            min=0)


    comp_mod.set_param_hint('c',
                            value=0.0,
                            vary=False)

    pars = comp_mod.make_params()

    return [comp_mod,pars,min_index,max_index]