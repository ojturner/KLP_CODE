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
import pyregion
from astropy.stats import median_absolute_deviation as astmad
from scipy.spatial import distance
from scipy import ndimage
from copy import copy
from lmfit import Model
from itertools import cycle as cycle
from spectral_cube import SpectralCube
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
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import Gaussian1DKernel

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
import klp_spaxel_fitting as klp_spax_fit
import klp_integrated_spectrum_fitting as int_spec_fit

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

def fit_all_regions(regions_file,
                    yj_name,
                    h_name,
                    k_name,
                    redshift,
                    plot_save_dir,
                    prog,
                    weight=False,
                    spatial_smooth=False,
                    smoothing_psf=0.4,
                    spectral_smooth=False,
                    spectral_smooth_width=1,
                    make_plot=True,
                    mcmc_output=False,
                    mcmc_cycles=500):

    """
    Def:
    Extract the pre-defined DS9 regions from the datacubes across
    the three wavebands. Fit both the continuum and the spectral
    lines using the most up to date fitting routines, constraining the
    [OIII]4960 flux to be 1/2.99 that of the [OIII]5007 flux and a double
    gaussian fit for the [OII] line.

    Input: regions_file
           yj_name
           h_name
           k_name
    Output: dictionary of the fit parameters including linewidths and fluxes
    """

    print '[INFO: ] FITTING %s' % yj_name
    # define the wave array in the yj,h and k bands
    yj_cube = cubeOps(yj_name)
    yj_cube_data = yj_cube.data
    yj_wave_array = yj_cube.wave_array
    h_cube = cubeOps(h_name)
    h_cube_data = h_cube.data
    h_wave_array = h_cube.wave_array
    k_cube = cubeOps(k_name)
    k_cube_data = k_cube.data
    k_wave_array = k_cube.wave_array
    # need to first get each of the table names and headers
    yj_table = fits.open(yj_name)
    regions_yj = pyregion.open(regions_file).as_imagecoord(yj_table[1].header)
    sig_field_yj = fits.open(yj_name[:-5] + '_oii_signal_field.fits')[0].data
    # make a plot in each waveband showing the position of the
    # integrated aperture relative to the (blurred) signal field
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.imshow(sig_field_yj,interpolation='nearest',vmin=0,vmax=6E-19)
    patch_list, artist_list = regions_yj.get_mpl_patches_texts()
    for p in patch_list:
        ax.add_patch(p)
    for t in artist_list:
        ax.add_artist(t)
    fig.savefig(plot_save_dir + 'yj_aperture.png')
    plt.close('all')
    h_table = fits.open(h_name)
    regions_h = pyregion.open(regions_file).as_imagecoord(h_table[1].header)
    sig_field_h = fits.open(h_name[:-5] + '_oiii_signal_field.fits')[0].data
    # make a plot in each waveband showing the position of the
    # integrated aperture relative to the (blurred) signal field
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.imshow(sig_field_h,interpolation='nearest',vmin=0,vmax=6E-19)
    patch_list, artist_list = regions_h.get_mpl_patches_texts()
    for p in patch_list:
        ax.add_patch(p)
    for t in artist_list:
        ax.add_artist(t)
    fig.savefig(plot_save_dir + 'h_aperture.png')
    plt.close('all')
    k_table = fits.open(k_name)
    regions_k = pyregion.open(regions_file).as_imagecoord(k_table[1].header)
    sig_field_k = fits.open(k_name[:-5] + '_ha_signal_field.fits')[0].data
    # make a plot in each waveband showing the position of the
    # integrated aperture relative to the (blurred) signal field
    fig, ax = plt.subplots(1,1,figsize=(12,12))
    ax.imshow(sig_field_k,interpolation='nearest',vmin=0,vmax=6E-19)
    patch_list, artist_list = regions_k.get_mpl_patches_texts()
    for p in patch_list:
        ax.add_patch(p)
    for t in artist_list:
        ax.add_artist(t)
    fig.savefig(plot_save_dir + 'k_aperture.png')
    plt.close('all')

    # define the spectral cubes that will be extracted from in each band
    # also define the hdus required to define the masks
    yj_spectral_cube = SpectralCube.read(yj_name,hdu=1)
    yj_hdu = fits.PrimaryHDU(np.nanmedian(yj_table[1].data[300:1600,:,:],axis=0))
    h_spectral_cube = SpectralCube.read(h_name,hdu=1)
    h_hdu = fits.PrimaryHDU(np.nanmedian(h_table[1].data[300:1600,:,:],axis=0))
    k_spectral_cube = SpectralCube.read(k_name,hdu=1)
    k_hdu = fits.PrimaryHDU(np.nanmedian(k_table[1].data[300:1600,:,:],axis=0))

    # here we'll manipulate the spectral cubes depending on whether
    # spectral/spatial smoothing has been selected
    if spatial_smooth and spectral_smooth:

        print '[INFO:  ] smoothing data spatially with width %s filter' % smoothing_psf
        print '[INFO:  ] smoothing data spectrally with width %s filter' % spectral_smooth_width

        spatial_kernel = Gaussian2DKernel(stddev=smoothing_psf)
        spectral_kernel = Gaussian1DKernel(stddev=spectral_smooth_width)
        yj_spectral_cube = yj_spectral_cube.spatial_smooth(spatial_kernel)
        yj_spectral_cube = yj_spectral_cube.spectral_smooth(spectral_kernel)
        h_spectral_cube = h_spectral_cube.spatial_smooth(spatial_kernel)
        h_spectral_cube = h_spectral_cube.spectral_smooth(spectral_kernel)
        k_spectral_cube = k_spectral_cube.spatial_smooth(spatial_kernel)
        k_spectral_cube = k_spectral_cube.spectral_smooth(spectral_kernel)
        # the error spectrum depends sensitively on the smoothing
        # applied. If we want to smooth both spatially and spectrally
        # this reduces the noise/weights array massively
        # define the skyline weights array
        yj_weights_file = yj_name[:-5] + '_error_spectrum0' + str(smoothing_psf) + str(spectral_smooth_width) + '.fits'
        h_weights_file = h_name[:-5] + '_error_spectrum0' + str(smoothing_psf) + str(spectral_smooth_width) + '.fits'
        k_weights_file = k_name[:-5] + '_error_spectrum0' + str(smoothing_psf) + str(spectral_smooth_width) + '.fits'

    elif spatial_smooth and not(spectral_smooth):

        print '[INFO:  ] smoothing data spatially with %s filter' % smoothing_psf

        spatial_kernel = Gaussian2DKernel(stddev=smoothing_psf)
        yj_spectral_cube = yj_spectral_cube.spatial_smooth(spatial_kernel)
        h_spectral_cube = h_spectral_cube.spatial_smooth(spatial_kernel)
        k_spectral_cube = k_spectral_cube.spatial_smooth(spatial_kernel)

        # the error spectrum depends sensitively on the smoothing
        # applied. If we want to smooth both spatially and spectrally
        # this reduces the noise/weights array massively
        # define the skyline weights array
        yj_weights_file = yj_name[:-5] + '_error_spectrum0' + str(smoothing_psf) + '.fits'
        h_weights_file = h_name[:-5] + '_error_spectrum0' + str(smoothing_psf) + '.fits'
        k_weights_file = k_name[:-5] + '_error_spectrum0' + str(smoothing_psf) + '.fits'

    elif spectral_smooth and not(spatial_smooth):

        print '[INFO:  ] smoothing data spectrally with %s filter' % spectral_smooth_width

        spectral_kernel = Gaussian1DKernel(stddev=spectral_smooth_width)
        yj_spectral_cube = yj_spectral_cube.spectral_smooth(spectral_kernel)
        h_spectral_cube = h_spectral_cube.spectral_smooth(spectral_kernel)
        k_spectral_cube = k_spectral_cube.spectral_smooth(spectral_kernel)

        # define the weights file
        yj_weights_file = yj_name[:-5] + '_error_spectrum' + str(spectral_smooth_width) + '.fits'
        h_weights_file = h_name[:-5] + '_error_spectrum' + str(spectral_smooth_width) + '.fits'
        yj_weights_file = k_name[:-5] + '_error_spectrum' + str(spectral_smooth_width) + '.fits'

    else:

        yj_weights_file = yj_name[:-5] + '_error_spectrum.fits'
        h_weights_file = h_name[:-5] + '_error_spectrum.fits'
        k_weights_file = k_name[:-5] + '_error_spectrum.fits'

        print '[INFO:  ] No data smoothing selected'

    # and readin/initiate the weights
    yj_weights = fits.open(yj_weights_file)[0].data
    h_weights = fits.open(h_weights_file)[0].data
    k_weights = fits.open(k_weights_file)[0].data

    # initiate the loop to extract the spectra from the same region 
    # in each waveband
    plot_counter = 0

    # also create the bpt arrays
    klp_log_oiii_hb = []
    klp_lower_error_log_oiii_hb = []
    klp_upper_error_log_oiii_hb = []
    klp_log_nii_ha = []
    klp_lower_error_log_nii_ha = []
    klp_upper_error_log_nii_ha = []

    for yj_reg, h_reg, k_reg in zip(regions_yj, regions_h, regions_k):

        yj_mask = pyregion.get_mask(pyregion.ShapeList([yj_reg]),yj_hdu)
        yj_number_pixels = yj_mask.sum()
        yj_masked_cube = yj_spectral_cube.with_mask(yj_mask)
        yj_spectrum  = np.array(yj_masked_cube.sum(axis=(1,2)))

        ################################################################
        ######CODE CHUNK TO CALCULATE INTEGRATED SPECTRUM NOISE#########
        # calculate the factor by which the weights must be scaled
        # empirically
        ################################################################

        yj_spaxel_array = []
        for i in range(yj_mask.shape[0]):
            for j in range(yj_mask.shape[1]):
                if yj_mask[i,j]:
                    yj_spaxel_array.append(yj_cube_data[:,i,j])
        yj_dispersion_array = []

        # for each of these spectra find the dispersion as a single number
        # after first masking the skylines and the emission lines

        for entry in yj_spaxel_array:

            yj_wave_array, yj_sky_masked_spec = mask_the_sky.masking_sky(yj_wave_array,
                                                                         entry,
                                                                         'YJ')

            # now mask the emission lines

            yj_sky_and_emission_line_masked_spec = klp_spax_fit.mask_emission_lines(yj_wave_array,
                                                                                    yj_sky_masked_spec,
                                                                                    redshift,
                                                                                    'YJ',
                                                                                    prog)

            yj_dispersion_array.append(astmad(yj_sky_and_emission_line_masked_spec[200:1800],ignore_nan=True))

        # the single spectrum dispersion value is the median of this array
        yj_single_dispersion = np.nanmedian(yj_dispersion_array)

        # now repeat the process for the summed spectrum

        yj_wave_array, yj_sky_masked_spec = mask_the_sky.masking_sky(yj_wave_array,
                                                                     yj_spectrum,
                                                                     'YJ')

        # now mask the emission lines

        yj_sky_and_emission_line_masked_spec = klp_spax_fit.mask_emission_lines(yj_wave_array,
                                                                                yj_sky_masked_spec,
                                                                                redshift,
                                                                                'YJ',
                                                                                prog)

        yj_summed_dispersion = astmad(yj_sky_and_emission_line_masked_spec[200:1800],ignore_nan=True)
        weights_increase_factor = (yj_summed_dispersion/yj_single_dispersion)

        print 'THIS IS THE YJ IDEAL CASE FACTOR: %s' % (np.sqrt(yj_number_pixels))
        print 'THIS IS THE YJ UPPER LIMIT INCREASE FACTOR: %s' % yj_number_pixels
        print 'THIS IS THE YJ ACTUAL FACTOR: %s' % (yj_summed_dispersion/yj_single_dispersion)

        if np.isnan(weights_increase_factor):
            weights_increase_factor = 1
        yj_weights_new = weights_increase_factor*yj_weights

        #############################################################################
        ##########END OF DISPERSION CALCULATING CODE CHUNK###########################
        #############################################################################

        yj_poly = int_spec_fit.continuum_subtract(yj_spectrum,
                                                  yj_wave_array,
                                                  'YJ',
                                                  redshift,
                                                  100,
                                                  1900,
                                                  prog)
        yj_spectrum[100:1900] = yj_spectrum[100:1900] - yj_poly[100:1900]

        # also want to get the name of the galaxy
        if yj_name == -1:
            yj_gal_name = copy(yj_name)
        # Otherwise the directory structure is included and have to
        # search for the backslash and omit up to the last one
        else:
            yj_gal_name = yj_name[len(yj_name) - yj_name[::-1].find("/"):]

        yj_gal_name = yj_gal_name[:-5]
        #yj_plot_save_name = plot_save_dir + 'yj_' + str(plot_counter) + '_fit.png'
        yj_plot_save_name = plot_save_dir + 'yj_integrated_fit.png'

        # if the mcmc output option is specified - calculate the line fluxes
        # and velocity dispersions of the lines using many hundreds of fits
        # this also provides the errors on the integrated emission lines
        # and will allow us to construct the final models as nice plots
        # at the end
        np.set_printoptions(threshold=np.nan)

        if mcmc_output:
            # setup lists to house the fit parameters 
            # loop over the number of times the MCMC if specified for
            oii_l_flux_array = []
            oii_l_sn_array = []
            oii_l_centre_array = []
            oii_h_flux_array = []
            oii_h_sn_array = []
            oii_h_centre_array = []
            oii_sigma_array = []

            print '[INFO: ] Running YJband MCMC'

            for i in range(mcmc_cycles):

                # need to perturb the flux values each time by the weights
                # spectrum 

                perturbed_spec = perturb_value(yj_weights_new,
                                               yj_spectrum)
                yj_output_array = int_spec_fit.line_fit(yj_wave_array,
                                                        perturbed_spec,
                                                        yj_weights_new,
                                                        redshift,
                                                        'YJ',
                                                        yj_gal_name,
                                                        yj_plot_save_name,
                                                        masked=False,
                                                        prog=prog,
                                                        weight=weight,
                                                        make_plot=make_plot)

                # append the results to the appropriate arrays
                # the resulting distributions should be gaussian
                oii_l_flux_array.append(yj_output_array[0,0])
                oii_l_sn_array.append(yj_output_array[0,2])
                oii_l_centre_array.append(yj_output_array[0,5])
                oii_sigma_array.append(yj_output_array[0,3])
                oii_h_flux_array.append(yj_output_array[1,0])
                oii_h_sn_array.append(yj_output_array[1,2])
                oii_h_centre_array.append(yj_output_array[1,5])

            # now calculate histograms,central values and errors

            # oii_lower_amplitude
            oii_l_flux_array = np.array(oii_l_flux_array)
            oii_l_flux = np.nanmedian(oii_l_flux_array)
            oii_l_flux_lower_error = oii_l_flux - np.nanpercentile(oii_l_flux_array, 16)
            oii_l_flux_upper_error = np.nanpercentile(oii_l_flux_array, 84) - oii_l_flux
            # oii_lower_signal_to_noise
            oii_l_sn_array = np.array(oii_l_sn_array)
            oii_l_sn = np.nanmedian(oii_l_sn_array)
            oii_l_sn_lower_error = oii_l_sn - np.nanpercentile(oii_l_sn_array, 16)
            oii_l_sn_upper_error = np.nanpercentile(oii_l_sn_array, 84) - oii_l_sn
            # oii_higher_amplitude
            oii_h_flux_array = np.array(oii_h_flux_array)
            oii_h_flux = np.nanmedian(oii_h_flux_array)
            oii_h_flux_lower_error = oii_h_flux - np.nanpercentile(oii_h_flux_array, 16)
            oii_h_flux_upper_error = np.nanpercentile(oii_h_flux_array, 84) - oii_h_flux
            # oii_higher_amplitude
            oii_h_sn_array = np.array(oii_h_sn_array)
            oii_h_sn = np.nanmedian(oii_h_sn_array)
            oii_h_sn_lower_error = oii_h_sn - np.nanpercentile(oii_h_sn_array, 16)
            oii_h_sn_upper_error = np.nanpercentile(oii_h_sn_array, 84) - oii_h_sn
            # oii_sigma
            oii_sigma_array = np.array(oii_sigma_array)
            oii_sigma = np.nanmedian(oii_sigma_array)
            oii_sigma_lower_error = oii_sigma - np.nanpercentile(oii_sigma_array, 16)
            oii_sigma_upper_error = np.nanpercentile(oii_sigma_array, 84) - oii_sigma
            # oii_higher_centre
            oii_h_centre_array = np.array(oii_h_centre_array)
            oii_h_centre = np.nanmedian(oii_h_centre_array)
            oii_h_centre_lower_error = oii_h_centre - np.nanpercentile(oii_h_centre_array, 16)
            oii_h_centre_upper_error = np.nanpercentile(oii_h_centre_array, 84) - oii_h_centre
            # oii_lower_centre
            oii_l_centre_array = np.array(oii_l_centre_array)
            oii_l_centre = np.nanmedian(oii_l_centre_array)
            oii_l_centre_lower_error = oii_l_centre - np.nanpercentile(oii_l_centre_array, 16)
            oii_l_centre_upper_error = np.nanpercentile(oii_l_centre_array, 84) - oii_l_centre

        else:
            yj_output_array = int_spec_fit.line_fit(yj_wave_array,
                                                    yj_spectrum,
                                                    yj_weights_new,
                                                    redshift,
                                                    'YJ',
                                                    yj_gal_name,
                                                    yj_plot_save_name,
                                                    masked=False,
                                                    prog=prog,
                                                    weight=weight,
                                                    make_plot=make_plot)

            oii_sigma = yj_output_array[0,3]
            oii_sigma_lower_error = yj_output_array[0,4]
            oii_sigma_upper_error = yj_output_array[0,4]
            oii_l_flux = yj_output_array[0,0]
            oii_l_flux_lower_error = yj_output_array[0,1]
            oii_l_flux_upper_error = yj_output_array[0,1]
            oii_l_sn = yj_output_array[0,2]
            oii_l_sn_lower_error = 0.0
            oii_l_sn_upper_error = 0.0
            oii_h_flux = yj_output_array[1,0]
            oii_h_flux_lower_error = yj_output_array[1,1]
            oii_h_flux_upper_error = yj_output_array[1,1]
            oii_h_sn = yj_output_array[1,2]
            oii_h_sn_lower_error = 0.0
            oii_h_sn_upper_error = 0.0
            oii_l_centre = yj_output_array[0,5]
            oii_h_centre = yj_output_array[1,5]

        # working out the error on the OII flux
        oii_total_flux = oii_l_flux + oii_h_flux
        oii_total_flux_lower_error = np.sqrt(oii_l_flux_lower_error**2 + oii_h_flux_lower_error**2)
        oii_total_flux_upper_error = np.sqrt(oii_l_flux_upper_error**2 + oii_h_flux_upper_error**2)

        # calculating the sn value to use for the total OII flux
        max_oii_sn = np.nanmax([oii_l_sn,oii_h_sn])
        max_oii_flux = np.nanmax([oii_l_flux,oii_h_flux])
        norm_oii_l_flux = oii_l_flux / max_oii_flux
        norm_oii_h_flux = oii_h_flux / max_oii_flux
        oii_sn_scaling_factor = np.sqrt(norm_oii_l_flux+norm_oii_h_flux)
        oii_total_sn = oii_sn_scaling_factor*max_oii_sn
        # check which oii_line the max sn comes from
        oii_max_ind = np.nanargmax([oii_l_sn,oii_h_sn])
        if oii_max_ind == 0:
            oii_total_sn_lower_error = oii_sn_scaling_factor*oii_l_sn_lower_error
            oii_total_sn_upper_error = oii_sn_scaling_factor*oii_l_sn_upper_error
        else:
            oii_total_sn_lower_error = oii_sn_scaling_factor*oii_h_sn_lower_error
            oii_total_sn_upper_error = oii_sn_scaling_factor*oii_h_sn_upper_error   

        h_mask = pyregion.get_mask(pyregion.ShapeList([h_reg]),h_hdu)
        h_number_pixels = h_mask.sum()
        h_masked_cube = h_spectral_cube.with_mask(h_mask)
        h_spectrum  = np.array(h_masked_cube.sum(axis=(1,2)))

        ################################################################
        ######CODE CHUNK TO CALCULATE INTEGRATED SPECTRUM NOISE#########
        # calculate the factor by which the weights must be scaled
        # empirically
        ################################################################

        h_spaxel_array = []
        for i in range(h_mask.shape[0]):
            for j in range(h_mask.shape[1]):
                if h_mask[i,j]:
                    h_spaxel_array.append(h_cube_data[:,i,j])
        h_dispersion_array = []

        # for each of these spectra find the dispersion as a single number
        # after first masking the skylines and the emission lines

        for entry in h_spaxel_array:

            h_wave_array, h_sky_masked_spec = mask_the_sky.masking_sky(h_wave_array,
                                                                         entry,
                                                                         'H')

            # now mask the emission lines

            h_sky_and_emission_line_masked_spec = klp_spax_fit.mask_emission_lines(h_wave_array,
                                                                                    h_sky_masked_spec,
                                                                                    redshift,
                                                                                    'H',
                                                                                    prog)

            h_dispersion_array.append(astmad(h_sky_and_emission_line_masked_spec[200:1800],ignore_nan=True))

        # the single spectrum dispersion value is the median of this array
        h_single_dispersion = np.nanmedian(h_dispersion_array)

        # now repeat the process for the summed spectrum

        h_wave_array, h_sky_masked_spec = mask_the_sky.masking_sky(h_wave_array,
                                                                     h_spectrum,
                                                                     'H')

        # now mask the emission lines

        h_sky_and_emission_line_masked_spec = klp_spax_fit.mask_emission_lines(h_wave_array,
                                                                                h_sky_masked_spec,
                                                                                redshift,
                                                                                'H',
                                                                                prog)

        h_summed_dispersion = astmad(h_sky_and_emission_line_masked_spec[200:1800],ignore_nan=True)
        weights_increase_factor = (h_summed_dispersion/h_single_dispersion)

        print 'THIS IS THE H IDEAL CASE FACTOR: %s' % (np.sqrt(h_number_pixels))
        print 'THIS IS THE H UPPER LIMIT INCREASE FACTOR: %s' % h_number_pixels
        print 'THIS IS THE H ACTUAL FACTOR: %s' % (h_summed_dispersion/h_single_dispersion)

        h_weights_new = weights_increase_factor*h_weights

        #############################################################################
        ##########END OF DISPERSION CALCULATING CODE CHUNK###########################
        #############################################################################

        h_poly = int_spec_fit.continuum_subtract(h_spectrum,
                                                 h_wave_array,
                                                 'H',
                                                 redshift,
                                                 100,
                                                 1700,
                                                 prog)
        h_spectrum[100:1700] = h_spectrum[100:1700] - h_poly[100:1700]

        # also want to get the name of the galaxy
        if h_name == -1:
            h_gal_name = copy(h_name)
        # Otherwise the directory structure is included and have to
        # search for the backslash and omit up to the last one
        else:
            h_gal_name = h_name[len(h_name) - h_name[::-1].find("/"):]

        h_gal_name = h_gal_name[:-5]
        #h_plot_save_name = plot_save_dir + 'h_' + str(plot_counter) + '_fit.png'
        h_plot_save_name = plot_save_dir + 'h_integrated_fit.png'

        # do the same thing with the H-band and MCMC fitting
        if mcmc_output:
            # setup lists to house the fit parameters 
            # loop over the number of times the MCMC if specified for
            hb_flux_array = []
            hb_sn_array = []
            hb_centre_array = []
            oiii4_flux_array = []
            oiii4_sn_array = []
            oiii4_centre_array = []
            oiii5_flux_array = []
            oiii5_sn_array = []
            oiii5_centre_array = []
            oiii5_sigma_array = []

            print '[INFO: ] Running Hband MCMC'

            for i in range(mcmc_cycles):

                # need to perturb the flux values each time by the weights
                # spectrum 

                perturbed_spec = perturb_value(h_weights_new,
                                               h_spectrum)
                h_output_array = int_spec_fit.line_fit(h_wave_array,
                                                       perturbed_spec,
                                                       h_weights_new,
                                                       redshift,
                                                       'H',
                                                       h_gal_name,
                                                       h_plot_save_name,
                                                       masked=False,
                                                       prog=prog,
                                                       weight=weight,
                                                       make_plot=make_plot)

                hb_flux_array.append(h_output_array[0,0])
                hb_sn_array.append(h_output_array[0,2])
                hb_centre_array.append(h_output_array[0,5])
                oiii4_flux_array.append(h_output_array[2,0])
                oiii4_sn_array.append(h_output_array[2,2])
                oiii4_centre_array.append(h_output_array[2,5])
                oiii5_flux_array.append(h_output_array[1,0])
                oiii5_sn_array.append(h_output_array[1,2])
                oiii5_centre_array.append(h_output_array[1,5])
                oiii5_sigma_array.append(h_output_array[1,3])

            # now calculate histograms,central values and errors

            # Hb_amplitude
            hb_flux_array = np.array(hb_flux_array)
            hb_flux = np.nanmedian(hb_flux_array)
            hb_flux_lower_error = hb_flux - np.nanpercentile(hb_flux_array, 16)
            hb_flux_upper_error = np.nanpercentile(hb_flux_array, 84) - hb_flux
            # Hb_sn
            hb_sn_array = np.array(hb_sn_array)
            hb_sn = np.nanmedian(hb_sn_array)
            hb_sn_lower_error = hb_sn - np.nanpercentile(hb_sn_array, 16)
            hb_sn_upper_error = np.nanpercentile(hb_sn_array, 84) - hb_sn
            # Hb_centre
            hb_centre_array = np.array(hb_centre_array)
            hb_centre = np.nanmedian(hb_centre_array)
            hb_centre_lower_error = hb_centre - np.nanpercentile(hb_centre_array, 16)
            hb_centre_upper_error = np.nanpercentile(hb_centre_array, 84) - hb_centre
            # oiii4 amplitude
            oiii4_flux_array = np.array(oiii4_flux_array)
            oiii4_flux = np.nanmedian(oiii4_flux_array)
            oiii4_flux_lower_error = oiii4_flux - np.nanpercentile(oiii4_flux_array, 16)
            oiii4_flux_upper_error = np.nanpercentile(oiii4_flux_array, 84) - oiii4_flux
            # oiii4 sn
            oiii4_sn_array = np.array(oiii4_sn_array)
            oiii4_sn = np.nanmedian(oiii4_sn_array)
            oiii4_sn_lower_error = oiii4_sn - np.nanpercentile(oiii4_sn_array, 16)
            oiii4_sn_upper_error = np.nanpercentile(oiii4_sn_array, 84) - oiii4_sn
            # oiii4 centre
            oiii4_centre_array = np.array(oiii4_centre_array)
            oiii4_centre = np.nanmedian(oiii4_centre_array)
            oiii4_centre_lower_error = oiii4_centre - np.nanpercentile(oiii4_centre_array, 16)
            oiii4_centre_upper_error = np.nanpercentile(oiii4_centre_array, 84) - oiii4_centre
            # oiii5 amplitude
            oiii5_flux_array = np.array(oiii5_flux_array)
            oiii5_flux = np.nanmedian(oiii5_flux_array)
            oiii5_flux_lower_error = oiii5_flux - np.nanpercentile(oiii5_flux_array, 16)
            oiii5_flux_upper_error = np.nanpercentile(oiii5_flux_array, 84) - oiii5_flux
            # oiii5 amplitude
            oiii5_sn_array = np.array(oiii5_sn_array)
            oiii5_sn = np.nanmedian(oiii5_sn_array)
            oiii5_sn_lower_error = oiii5_sn - np.nanpercentile(oiii5_sn_array, 16)
            oiii5_sn_upper_error = np.nanpercentile(oiii5_sn_array, 84) - oiii5_sn
            # oiii5 sigma
            oiii5_sigma_array = np.array(oiii5_sigma_array)
            oiii5_sigma = np.nanmedian(oiii5_sigma_array)
            oiii5_sigma_lower_error = oiii5_sigma - np.nanpercentile(oiii5_sigma_array, 16)
            oiii5_sigma_upper_error = np.nanpercentile(oiii5_sigma_array, 84) - oiii5_sigma
            # oiii5 centre
            oiii5_centre_array = np.array(oiii5_centre_array)
            oiii5_centre = np.nanmedian(oiii5_centre_array)
            oiii5_centre_lower_error = oiii5_centre - np.nanpercentile(oiii5_centre_array, 16)
            oiii5_centre_upper_error = np.nanpercentile(oiii5_centre_array, 84) - oiii5_centre

        else:

            h_output_array = int_spec_fit.line_fit(h_wave_array,
                                                   h_spectrum,
                                                   h_weights_new,
                                                   redshift,
                                                   'H',
                                                   h_gal_name,
                                                   h_plot_save_name,
                                                   masked=False,
                                                   prog=prog,
                                                   weight=weight,
                                                   make_plot=make_plot)

            hb_flux = h_output_array[0,0]
            hb_flux_lower_error = h_output_array[0,1]
            hb_flux_upper_error = h_output_array[0,1]
            hb_sn = h_output_array[0,2]
            hb_sn_lower_error = 0.0
            hb_sn_upper_error = 0.0
            hb_centre = h_output_array[0,5]
            oiii4_flux = h_output_array[2,0]
            oiii4_flux_lower_error = h_output_array[2,1]
            oiii4_flux_upper_error = h_output_array[2,1]
            oiii4_sn = h_output_array[2,2]
            oiii4_sn_lower_error = 0.0
            oiii4_sn_upper_error = 0.0
            oiii4_centre = h_output_array[2,5]
            oiii5_flux = h_output_array[1,0]
            oiii5_flux_lower_error = h_output_array[1,1]
            oiii5_flux_upper_error = h_output_array[1,1]
            oiii5_sn = h_output_array[1,2]
            oiii5_sn_lower_error = 0.0
            oiii5_sn_upper_error = 0.0
            oiii5_sigma = h_output_array[1,3]
            oiii5_sigma_lower_error = h_output_array[1,4]
            oiii5_sigma_upper_error = h_output_array[1,4]
            oiii5_centre = h_output_array[1,5]

        k_mask = pyregion.get_mask(pyregion.ShapeList([k_reg]),k_hdu)
        k_number_pixels = k_mask.sum()
        k_masked_cube = k_spectral_cube.with_mask(k_mask)
        k_spectrum  = np.array(k_masked_cube.sum(axis=(1,2)))

        ################################################################
        ######CODE CHUNK TO CALCULATE INTEGRATED SPECTRUM NOISE#########
        # calculate the factor by which the weights must be scaled
        # empirically
        ################################################################

        k_spaxel_array = []
        for i in range(k_mask.shape[0]):
            for j in range(k_mask.shape[1]):
                if k_mask[i,j]:
                    k_spaxel_array.append(k_cube_data[:,i,j])
        k_dispersion_array = []

        # for each of these spectra find the dispersion as a single number
        # after first masking the skylines and the emission lines

        for entry in k_spaxel_array:

            k_wave_array, k_sky_masked_spec = mask_the_sky.masking_sky(k_wave_array,
                                                                         entry,
                                                                         'K')

            # now mask the emission lines

            k_sky_and_emission_line_masked_spec = klp_spax_fit.mask_emission_lines(k_wave_array,
                                                                                    k_sky_masked_spec,
                                                                                    redshift,
                                                                                    'K',
                                                                                    prog)

            k_dispersion_array.append(astmad(k_sky_and_emission_line_masked_spec[200:1800],ignore_nan=True))

        # the single spectrum dispersion value is the median of this array
        k_single_dispersion = np.nanmedian(k_dispersion_array)

        # now repeat the process for the summed spectrum

        k_wave_array, k_sky_masked_spec = mask_the_sky.masking_sky(k_wave_array,
                                                                     k_spectrum,
                                                                     'K')

        # now mask the emission lines

        k_sky_and_emission_line_masked_spec = klp_spax_fit.mask_emission_lines(k_wave_array,
                                                                                k_sky_masked_spec,
                                                                                redshift,
                                                                                'K',
                                                                                prog)

        k_summed_dispersion = astmad(k_sky_and_emission_line_masked_spec[200:1800],ignore_nan=True)
        weights_increase_factor = (k_summed_dispersion/k_single_dispersion)

        print 'THIS IS THE K IDEAL CASE FACTOR: %s' % (np.sqrt(k_number_pixels))
        print 'THIS IS THE K UPPER LIMIT INCREASE FACTOR: %s' % k_number_pixels
        print 'THIS IS THE K ACTUAL FACTOR: %s' % (k_summed_dispersion/k_single_dispersion)

        k_weights_new = weights_increase_factor*k_weights

        #############################################################################
        ##########END OF DISPERSION CALCULATING CODE CHUNK###########################
        #############################################################################

        k_poly = int_spec_fit.continuum_subtract(k_spectrum,
                                                 k_wave_array,
                                                 'K',
                                                 redshift,
                                                 100,
                                                 1700,
                                                 prog)
        k_spectrum[100:1700] = k_spectrum[100:1700] - k_poly[100:1700]

        # also want to get the name of the galaxy
        if k_name == -1:
            k_gal_name = copy(k_name)
        # Otherwise the directory structure is included and have to
        # search for the backslash and omit up to the last one
        else:
            k_gal_name = k_name[len(k_name) - k_name[::-1].find("/"):]
        k_gal_name = k_gal_name[:-5]
        #k_plot_save_name = plot_save_dir + 'k_' + str(plot_counter) + '_fit.png'
        k_plot_save_name = plot_save_dir + 'k_integrated_fit.png'

        # do the same thing with the K-band and MCMC fitting
        if mcmc_output:
            # setup lists to house the fit parameters 
            # loop over the number of times the MCMC if specified for
            nii_flux_array = []
            nii_sn_array = []
            nii_centre_array = []
            ha_flux_array = []
            ha_sn_array = []
            ha_sigma_array = []
            ha_centre_array = []

            print '[INFO: ] Running Kband MCMC'

            for i in range(mcmc_cycles):

                # need to perturb the flux values each time by the weights
                # spectrum 

                perturbed_spec = perturb_value(k_weights_new,
                                               k_spectrum)
                k_output_array = int_spec_fit.line_fit(k_wave_array,
                                                       perturbed_spec,
                                                       k_weights_new,
                                                       redshift,
                                                       'K',
                                                       k_gal_name,
                                                       k_plot_save_name,
                                                       masked=False,
                                                       prog=prog,
                                                       weight=weight,
                                                       make_plot=make_plot)

                nii_flux_array.append(k_output_array[1,0])
                nii_sn_array.append(k_output_array[1,2])
                nii_centre_array.append(k_output_array[1,5])
                ha_flux_array.append(k_output_array[0,0])
                ha_sn_array.append(k_output_array[0,2])
                ha_sigma_array.append(k_output_array[0,3])
                ha_centre_array.append(k_output_array[0,5])

            # find the central values and errors by fitting histograms
            # nii amplitude
            nii_flux_array = np.array(nii_flux_array)
            nii_flux = np.nanmedian(nii_flux_array)
            nii_flux_lower_error = nii_flux - np.nanpercentile(nii_flux_array, 16)
            nii_flux_upper_error = np.nanpercentile(nii_flux_array, 84) - nii_flux
            # nii signal to noise
            nii_sn_array = np.array(nii_sn_array)
            nii_sn = np.nanmedian(nii_sn_array)
            nii_sn_lower_error = nii_sn - np.nanpercentile(nii_sn_array, 16)
            nii_sn_upper_error = np.nanpercentile(nii_sn_array, 84) - nii_sn
            # nii centre
            nii_centre_array = np.array(nii_centre_array)
            nii_centre = np.nanmedian(nii_centre_array)
            nii_centre_lower_error = nii_centre - np.nanpercentile(nii_centre_array, 16)
            nii_centre_upper_error = np.nanpercentile(nii_centre_array, 84) - nii_centre
            # ha amplitude
            ha_flux_array = np.array(ha_flux_array)
            ha_flux = np.nanmedian(ha_flux_array)
            ha_flux_lower_error = ha_flux - np.nanpercentile(ha_flux_array, 16)
            ha_flux_upper_error = np.nanpercentile(ha_flux_array, 84) - ha_flux
            # ha amplitude
            ha_sn_array = np.array(ha_sn_array)
            ha_sn = np.nanmedian(ha_sn_array)
            ha_sn_lower_error = ha_sn - np.nanpercentile(ha_sn_array, 16)
            ha_sn_upper_error = np.nanpercentile(ha_sn_array, 84) - ha_sn
            # ha sigma
            ha_sigma_array = np.array(ha_sigma_array)
            ha_sigma = np.nanmedian(ha_sigma_array)
            ha_sigma_lower_error = ha_sigma - np.nanpercentile(ha_sigma_array, 16)
            ha_sigma_upper_error = np.nanpercentile(ha_sigma_array, 84) - ha_sigma
            # ha centre
            ha_centre_array = np.array(ha_centre_array)
            ha_centre = np.nanmedian(ha_centre_array)
            ha_centre_lower_error = ha_centre - np.nanpercentile(ha_centre_array, 16)
            ha_centre_upper_error = np.nanpercentile(ha_centre_array, 84) - ha_centre

        else:

            k_output_array = int_spec_fit.line_fit(k_wave_array,
                                                   k_spectrum,
                                                   k_weights_new,
                                                   redshift,
                                                   'K',
                                                   k_gal_name,
                                                   k_plot_save_name,
                                                   masked=False,
                                                   prog=prog,
                                                   weight=weight,
                                                   make_plot=make_plot)

            nii_flux = k_output_array[1,0]
            nii_flux_lower_error = k_output_array[1,1]
            nii_flux_upper_error = k_output_array[1,1]
            nii_sn = k_output_array[1,2]
            nii_sn_lower_error = 0.0
            nii_sn_upper_error = 0.0
            nii_centre = k_output_array[1,5]
            ha_flux = k_output_array[0,0]
            ha_flux_lower_error = k_output_array[0,1]
            ha_flux_upper_error = k_output_array[0,1]
            ha_sn = k_output_array[0,2]
            ha_sn_lower_error = 0.0
            ha_sn_upper_error = 0.0
            ha_sigma = k_output_array[0,3]
            ha_sigma_lower_error = k_output_array[0,4]
            ha_sigma_upper_error = k_output_array[0,4]
            ha_centre = k_output_array[0,5]

        ################################################    
        ################################################
        # line ratios and error calculations
        ################################################
        ################################################

        # Calculating O32
        oiii_total_flux = oiii5_flux + oiii4_flux
        oiii_total_flux_lower_error = np.sqrt((oiii5_flux_lower_error)**2 + (oiii4_flux_lower_error)**2)
        oiii_total_flux_upper_error = np.sqrt((oiii5_flux_upper_error)**2 + (oiii4_flux_upper_error)**2)
        
        oiii_oii_ratio = (oiii_total_flux)/oii_total_flux
        oiii_oii_ratio_lower_error = oiii_oii_ratio*(np.sqrt((oiii_total_flux_lower_error/oiii_total_flux)**2 + (oii_total_flux_lower_error/oii_total_flux)**2))
        oiii_oii_ratio_upper_error = oiii_oii_ratio*(np.sqrt((oiii_total_flux_upper_error/oiii_total_flux)**2 + (oii_total_flux_upper_error/oii_total_flux)**2))

        # log oiii_oii
        log_oiii_oii = np.log10(oiii_oii_ratio)
        log_oiii_oii_lower_error = 0.434*(oiii_oii_ratio_lower_error/oiii_oii_ratio)
        log_oiii_oii_upper_error = 0.434*(oiii_oii_ratio_upper_error/oiii_oii_ratio)

        # Calculating R23
        r23_numerator = oiii_total_flux + oii_total_flux
        r23_numerator_lower_error = np.sqrt((oiii_total_flux_lower_error)**2 + (oii_total_flux_lower_error)**2)
        r23_numerator_upper_error = np.sqrt((oiii_total_flux_upper_error)**2 + (oii_total_flux_upper_error)**2)

        r23 = r23_numerator/hb_flux
        r23_lower_error = r23*(np.sqrt((r23_numerator_lower_error/r23_numerator)**2 + (hb_flux_lower_error/hb_flux)**2))
        r23_upper_error = r23*(np.sqrt((r23_numerator_upper_error/r23_numerator)**2 + (hb_flux_upper_error/hb_flux)**2))

        # log R23
        log_r23 = np.log10(r23)
        log_r23_lower_error = 0.434*(r23_lower_error/r23)
        log_r23_upper_error = 0.434*(r23_upper_error/r23)

        # now add to the bpt arrays
        # for the BPT calculations - use the average error? no
        # can calculate upper and lower errors using the proper values

        oiii_hb_ratio = oiii5_flux/hb_flux
        lower_error_oiii_hb_ratio = oiii_hb_ratio*(np.sqrt((oiii5_flux_lower_error/oiii5_flux)**2 + (hb_flux_lower_error/hb_flux)**2))
        upper_error_oiii_hb_ratio = oiii_hb_ratio*(np.sqrt((oiii5_flux_upper_error/oiii5_flux)**2 + (hb_flux_upper_error/hb_flux)**2))

        # take logarithm and find error on this
        log_oiii_hb = np.log10(oiii_hb_ratio)
        lower_error_log_oiii_hb = 0.434*(lower_error_oiii_hb_ratio/oiii_hb_ratio)
        upper_error_log_oiii_hb = 0.434*(upper_error_oiii_hb_ratio/oiii_hb_ratio)

        # append to the arrays created at the beginning of the script
        klp_log_oiii_hb.append(log_oiii_hb)
        klp_lower_error_log_oiii_hb.append(lower_error_log_oiii_hb)
        klp_upper_error_log_oiii_hb.append(upper_error_log_oiii_hb)

        nii_ha_ratio = nii_flux/ha_flux
        lower_error_nii_ha_ratio = nii_ha_ratio*(np.sqrt((nii_flux_lower_error/nii_flux)**2 + (ha_flux_lower_error/ha_flux)**2))
        upper_error_nii_ha_ratio = nii_ha_ratio*(np.sqrt((nii_flux_upper_error/nii_flux)**2 + (ha_flux_upper_error/ha_flux)**2))

        # take logarithm and find error on this
        log_nii_ha = np.log10(nii_ha_ratio)
        lower_error_log_nii_ha = 0.434*(lower_error_nii_ha_ratio/nii_ha_ratio)
        upper_error_log_nii_ha = 0.434*(upper_error_nii_ha_ratio/nii_ha_ratio)

        # append to the arrays created at the beginning of the script
        klp_log_nii_ha.append(log_nii_ha)
        klp_lower_error_log_nii_ha.append(lower_error_log_nii_ha)
        klp_upper_error_log_nii_ha.append(upper_error_log_nii_ha)

        # increment the plot counter

        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('font', weight='bold')
        rc('text', usetex=True)
        rc('axes', linewidth=2)
        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

        # KBSS-MOSFIRE measurements
        kbss_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/KBSS_MOSFIRE_LINE_RATIOS/steidel14_line_ratios.txt')
        kbss_oiii_hb = kbss_table['log([OIII]/Hb)']
        kbss_oiii_hb_lower_error = kbss_table['e_log([OIII]/Hb)']
        kbss_oiii_hb_upper_error = kbss_table['E_log([OIII]/Hb)']
        kbss_nii_ha = kbss_table['log([NII]/Ha)']
        kbss_nii_ha_lower_error = kbss_table['e_log([NII]/Ha)']
        kbss_nii_ha_upper_error = kbss_table['E_log([NII]/Ha)']


        # sdss flux measurements
        sdss_table = fits.open('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/SDSS_LINE_FLUXES/sdss_reliable_flux_measurements.fits')
        sdss_oiii_hb = sdss_table[1].data['log_oiii_hb']
        sdss_nii_ha = sdss_table[1].data['log_nii_ha']

        fig, ax = plt.subplots(1,1, figsize=(10,10))

        ax.tick_params(axis='both',
                       which='major',
                       labelsize=26,
                       length=12,
                       width=4)
        ax.tick_params(axis='both', 
                       which='minor',
                       labelsize=26,
                       length=6,
                       width=4)

        ax.set_xlabel(r'log([NII] $\lambda$6585/H$\alpha$)',
                      fontsize=30,
                      fontweight='bold',
                      labelpad=15)

        ax.set_ylabel(r'log([OIII] $\lambda$5008/H$\beta$)',
                      fontsize=30,
                      fontweight='bold',
                      labelpad=15)

        lines = {'linestyle': 'None'}
        plt.rc('lines', **lines)

        cm = plt.cm.get_cmap('Blues')

        # try creating contour plot instead

        xedges = np.arange(-1.6,0.6,0.01)
        yedges = np.arange(-1.1,1.1,0.01)

        H, xedges, yedges = np.histogram2d(sdss_nii_ha,sdss_oiii_hb,bins=[xedges,yedges])

        diff_x = xedges[1] - xedges[0]
        diff_y = yedges[1] - yedges[0]

        xedges = xedges[:-1] + (diff_x / 2.0)
        yedges = yedges[:-1] + (diff_y / 2.0)

        X,Y = np.meshgrid(xedges,yedges)

        CS_blue = ax.contour(X,Y,H.T,
                             levels=np.arange(5,150,5.0),
                             cmap=cm,
                             label='SDSS z < 0.2 reliable Brinchmann+06')

        [i.set_linewidth(4.0) for i in ax.spines.itervalues()]

        for entry in CS_blue.collections:
            plt.setp(entry,linewidth=2.75)

        # plotting the KBSS redshift 2 points
        kbss_errorbar = ax.errorbar(kbss_nii_ha,
                                    kbss_oiii_hb,
                                    ecolor='black',
                                    xerr=[kbss_nii_ha_lower_error,kbss_nii_ha_upper_error],
                                    yerr=[kbss_oiii_hb_lower_error,kbss_oiii_hb_upper_error],
                                    marker='o',
                                    markersize=3,
                                    markerfacecolor='green',
                                    markeredgecolor='green',
                                    markeredgewidth=2,
                                    capsize=2,
                                    elinewidth=2,
                                    alpha=0.75,
                                    label='KBSS-MOSFIRE: Steidel+14')
        # plotting the KBSS redshift 2 points
        klps_errorbar = ax.errorbar(klp_log_nii_ha[plot_counter],
                                    klp_log_oiii_hb[plot_counter],
                                    ecolor='black',
                                    xerr=np.array([[klp_lower_error_log_nii_ha[plot_counter],klp_upper_error_log_nii_ha[plot_counter]]]).T,
                                    yerr=np.array([[klp_lower_error_log_oiii_hb[plot_counter],klp_upper_error_log_oiii_hb[plot_counter]]]).T,
                                    marker='*',
                                    markersize=20,
                                    markerfacecolor='red',
                                    markeredgecolor='red',
                                    markeredgewidth=2,
                                    capsize=3,
                                    elinewidth=3,
                                    label='KLP GAL REG')
        ax.legend(loc='best',
                    prop={'size':18,'weight':'bold'},
                    frameon=False,
                    markerscale=1.2,
                    numpoints=1)
        ax.set_xlim(-1.7,0.0)
        ax.set_ylim(-0.8,1.1)
        fig.tight_layout()
        #plt.show()
        plt.close('all')

        plot_counter += 1

        # for each galaxy want to make a 3x3 spectrum grid joining together
        # the individual spectrum fits for each waveband. This will be exactly
        # equivalent to the plots we already made in the integrated spectrum
        # fit but instead will showcase everything together

        # make the plot save name
        all_plot_save_name = plot_save_dir + 'all_integrated_fit.png'
        # define speed of light for conversions back to gaussian model
        c = 2.99792458E5

        line_fig, line_ax = plt.subplots(3,3,figsize=(24,14),
                                         gridspec_kw = {'height_ratios':[2,1,1],
                                                        'wspace':0,
                                                        'hspace':0,})

        ##############################################################
        ##############################################################
        #########YJBAND ALL MODEL GRID################################
        ##############################################################
        ##############################################################
        # define the line wavelengths
        oii_3727_rest = 0.3727092
        oii_3727_shifted = (1 + redshift) * oii_3727_rest

        oii_3729_rest = 0.3729875
        oii_3729_shifted = (1 + redshift) * oii_3729_rest
        # get the yjband sky dictionary
        yj_sky_dict = mask_the_sky.ret_yj_sky()

        # find the fitting range
        fitting_range_limit = 0.015
        yj_min_index = np.argmin(abs(yj_wave_array - (oii_3727_shifted - fitting_range_limit)))
        yj_max_index = np.argmin(abs(yj_wave_array - (oii_3729_shifted + fitting_range_limit)))

        # define the models for each waveband
        # YJband model
        yj_band_model = GaussianModel(missing='drop',
                                      prefix='oiil_') + \
                        GaussianModel(missing='drop',
                                      prefix='oiih_')
        # use the fitted values to set the model parameters
        # and then evaluate over the full wavelength range and plot
        oii_model_sigma = (oii_l_centre * oii_sigma) / c
        yj_band_model.set_param_hint('oiil_amplitude',value=oii_l_flux)
        yj_band_model.set_param_hint('oiih_amplitude',value=oii_h_flux)
        yj_band_model.set_param_hint('oiil_sigma',value=oii_model_sigma)
        yj_band_model.set_param_hint('oiih_sigma',value=oii_model_sigma)
        yj_band_model.set_param_hint('oiil_center',value=oii_l_centre)
        yj_band_model.set_param_hint('oiih_center',value=oii_h_centre)
        yj_pars = yj_band_model.make_params()
        yj_evaluated_model = yj_band_model.eval(params=yj_pars,x=yj_wave_array)
        # now find the residuals
        yj_residuals = yj_spectrum - yj_evaluated_model
        # populate the plot with everything
        lines = {'linestyle': '-'}
        plt.rc('lines', **lines)
        line_ax[0][0].plot(yj_wave_array[100:1950],yj_spectrum[100:1950]/1E-14,drawstyle='steps-mid')
        line_ax[0][0].plot(yj_wave_array[100:1950],yj_evaluated_model[100:1950]/1E-14,color='red',drawstyle='steps-mid',lw=2)
        line_ax[1][0].plot(yj_wave_array[100:1950],yj_weights_new[100:1950]/1E-14,color='green',drawstyle='steps-mid',lw=2)
        line_ax[0][0].plot(yj_wave_array[100:1950],yj_evaluated_model[100:1950]/1E-14,color='red')
        line_ax[0][0].fill_between(yj_wave_array[100:1950],
                                   np.repeat(0,len(yj_wave_array[100:1950])),
                                   yj_evaluated_model[100:1950]/1E-14,
                                    facecolor='red',
                                    edgecolor='red',
                                    alpha=0.5)
        lines = {'linestyle': 'None'}
        plt.rc('lines', **lines)
        line_ax[2][0].errorbar(yj_wave_array[100:1950],
                               yj_residuals[100:1950]/1E-14,
                               ecolor='midnightblue',
                               yerr=[yj_weights_new[100:1950]/1E-14,yj_weights_new[100:1950]/1E-14],
                               marker='o',
                               markersize=5,
                               markerfacecolor='midnightblue',
                               markeredgecolor='midnightblue',
                               markeredgewidth=1,
                               capsize=1,
                               elinewidth=1)
        lines = {'linestyle': '--'}
        plt.rc('lines', **lines)
        line_ax[2][0].axhline(y=0,color='black')
        lines = {'linestyle': '-'}
        plt.rc('lines', **lines)
        line_ax[0][0].set_xlim(yj_wave_array[yj_min_index],yj_wave_array[yj_max_index])
        line_ax[1][0].set_xlim(yj_wave_array[yj_min_index],yj_wave_array[yj_max_index])
        line_ax[2][0].set_xlim(yj_wave_array[yj_min_index],yj_wave_array[yj_max_index])
        line_ax[2][0].set_ylim(-5,5)
        line_ax[1][0].set_ylim((np.nanmin(yj_weights_new[yj_min_index:yj_max_index]/1E-14)),
                               (np.nanmax(yj_weights_new[yj_min_index:yj_max_index]/1E-14)))
        for ranges in yj_sky_dict.values():
            line_ax[0][0].axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')
            line_ax[2][0].axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')
        yticks = line_ax[0][0].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        # tick parameters 
        line_ax[0][0].tick_params(axis='both',
                       which='major',
                       labelsize=26,
                       length=12,
                       width=4)
        line_ax[0][0].tick_params(axis='both',
                       which='minor',
                       labelsize=26,
                       length=6,
                       width=4)
        line_ax[0][0].minorticks_on()
        line_ax[0][0].yaxis.get_offset_text().set_fontsize(26)
        [i.set_linewidth(4.0) for i in line_ax[0][0].spines.itervalues()]
        # tick parameters 
        line_ax[1][0].tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=12,
                       width=4)
        line_ax[1][0].tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=6,
                       width=4)
        [i.set_linewidth(4.0) for i in line_ax[1][0].spines.itervalues()]
        line_ax[2][0].tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=12,
                       width=4)
        line_ax[2][0].tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=6,
                       width=4)
        [i.set_linewidth(4.0) for i in line_ax[2][0].spines.itervalues()]

        ##############################################################
        ##############################################################
        #########HBAND ALL MODEL GRID################################
        ##############################################################
        ##############################################################
        # line wavelengths
        h_beta_rest = 0.4862721
        h_beta_shifted = (1 + redshift) * h_beta_rest
        oiii_4960_rest = 0.4960295
        oiii_4960_shifted = (1 + redshift) * oiii_4960_rest
        oiii_5008_rest = 0.5008239
        oiii_5008_shifted = (1 + redshift) * oiii_5008_rest

        # get the hband sky dictionary
        h_sky_dict = mask_the_sky.ret_h_sky()

        # find the fitting range
        fitting_range_limit = 0.005
        h_min_index = np.argmin(abs(h_wave_array - (h_beta_shifted - fitting_range_limit)))
        h_max_index = np.argmin(abs(h_wave_array - (oiii_5008_shifted + fitting_range_limit)))

        # construct a composite gaussian model with prefix parameter names
        h_band_model = GaussianModel(missing='drop',
                                     prefix='hb_') + \
                       GaussianModel(missing='drop',
                                     prefix='oiii4_') + \
                       GaussianModel(missing='drop',
                                     prefix='oiii5_')
        # use the fitted values to set the model parameters
        # and then evaluate over the full wavelength range and plot
        oiii_model_sigma = (oiii5_centre*oiii5_sigma) / c
        hb_model_sigma = (hb_centre*oiii5_sigma) / c
        h_band_model.set_param_hint('hb_amplitude',value=hb_flux)
        h_band_model.set_param_hint('oiii4_amplitude',value=oiii4_flux)
        h_band_model.set_param_hint('oiii5_amplitude',value=oiii5_flux)
        h_band_model.set_param_hint('hb_sigma',value=hb_model_sigma)
        h_band_model.set_param_hint('oiii4_sigma',value=oiii_model_sigma)
        h_band_model.set_param_hint('oiii5_sigma',value=oiii_model_sigma)
        h_band_model.set_param_hint('hb_center',value=hb_centre)
        h_band_model.set_param_hint('oiii4_center',value=oiii4_centre)
        h_band_model.set_param_hint('oiii5_center',value=oiii5_centre)
        h_pars = h_band_model.make_params()
        h_evaluated_model = h_band_model.eval(params=h_pars,x=h_wave_array)
        # now find the residuals
        h_residuals = h_spectrum - h_evaluated_model
        # populate the plot with everything
        lines = {'linestyle': '-'}
        plt.rc('lines', **lines)
        line_ax[0][1].plot(h_wave_array[100:1950],h_spectrum[100:1950]/1E-14,drawstyle='steps-mid')
        line_ax[0][1].plot(h_wave_array[100:1950],h_evaluated_model[100:1950]/1E-14,color='red',drawstyle='steps-mid',lw=2)
        line_ax[1][1].plot(h_wave_array[100:1950],h_weights_new[100:1950]/1E-14,color='green',drawstyle='steps-mid',lw=2)
        line_ax[0][1].plot(h_wave_array[100:1950],h_evaluated_model[100:1950]/1E-14,color='red')
        line_ax[0][1].fill_between(h_wave_array[100:1950],
                                   np.repeat(0,len(h_wave_array[100:1950])),
                                   h_evaluated_model[100:1950]/1E-14,
                                    facecolor='red',
                                    edgecolor='red',
                                    alpha=0.5)
        lines = {'linestyle': 'None'}
        plt.rc('lines', **lines)
        line_ax[2][1].errorbar(h_wave_array[100:1950],
                               h_residuals[100:1950]/1E-14,
                               ecolor='midnightblue',
                               yerr=[h_weights_new[100:1950]/1E-14,h_weights_new[100:1950]/1E-14],
                               marker='o',
                               markersize=5,
                               markerfacecolor='midnightblue',
                               markeredgecolor='midnightblue',
                               markeredgewidth=1,
                               capsize=1,
                               elinewidth=1)
        lines = {'linestyle': '--'}
        plt.rc('lines', **lines)
        line_ax[2][1].axhline(y=0,color='black')
        lines = {'linestyle': '-'}
        plt.rc('lines', **lines)
        line_ax[0][1].set_xlim(h_wave_array[h_min_index],h_wave_array[h_max_index])
        line_ax[1][1].set_xlim(h_wave_array[h_min_index],h_wave_array[h_max_index])
        line_ax[2][1].set_xlim(h_wave_array[h_min_index],h_wave_array[h_max_index])
        line_ax[2][1].set_ylim(-5,5)
        line_ax[1][1].set_ylim((np.nanmin(h_weights_new[h_min_index:h_max_index]/1E-14)),
                               (np.nanmax(h_weights_new[h_min_index:h_max_index]/1E-14)))
        for ranges in h_sky_dict.values():
            line_ax[0][1].axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')
            line_ax[2][1].axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')
        yticks = line_ax[0][1].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        # tick parameters 
        line_ax[0][1].tick_params(axis='both',
                       which='major',
                       labelsize=26,
                       length=12,
                       width=4)
        line_ax[0][1].tick_params(axis='both',
                       which='minor',
                       labelsize=26,
                       length=6,
                       width=4)
        line_ax[0][1].minorticks_on()
        line_ax[0][1].yaxis.get_offset_text().set_fontsize(26)
        [i.set_linewidth(4.0) for i in line_ax[0][1].spines.itervalues()]
        # tick parameters 
        line_ax[1][1].tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=12,
                       width=4)
        line_ax[1][1].tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=6,
                       width=4)
        [i.set_linewidth(4.0) for i in line_ax[1][1].spines.itervalues()]
        line_ax[2][1].tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=12,
                       width=4)
        line_ax[2][1].tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=6,
                       width=4)
        [i.set_linewidth(4.0) for i in line_ax[2][1].spines.itervalues()]

        ##############################################################
        ##############################################################
        #########KBAND ALL MODEL GRID################################
        ##############################################################
        ##############################################################
        # line wavelengths
        h_alpha_rest = 0.6564614
        h_alpha_shifted = (1 + redshift) * h_alpha_rest
        nii_rest = 0.658523
        nii_shifted = (1 + redshift) * nii_rest

        # get the kband sky dictionary
        k_sky_dict = mask_the_sky.ret_k_sky()

        # find the fitting range
        fitting_range_limit = 0.015
        k_min_index = np.argmin(abs(k_wave_array - (h_alpha_shifted - fitting_range_limit)))
        k_max_index = np.argmin(abs(k_wave_array - (nii_shifted + fitting_range_limit)))

        # construct a composite gaussian model with prefix parameter names
        k_band_model = GaussianModel(missing='drop',
                                     prefix='ha_') + \
                       GaussianModel(missing='drop',
                                     prefix='nii_')
        # use the fitted values to set the model parameters
        # and then evaluate over the full wavelength range and plot

        ha_model_sigma = (ha_centre * ha_sigma) / c
        k_band_model.set_param_hint('ha_amplitude',value=ha_flux)
        k_band_model.set_param_hint('nii_amplitude',value=nii_flux)
        k_band_model.set_param_hint('ha_sigma',value=ha_model_sigma)
        k_band_model.set_param_hint('nii_sigma',value=ha_model_sigma)
        k_band_model.set_param_hint('ha_center',value=ha_centre)
        k_band_model.set_param_hint('nii_center',value=nii_centre)
        k_pars = k_band_model.make_params()
        k_evaluated_model = k_band_model.eval(params=k_pars,x=k_wave_array)
        # now find the residuals
        k_residuals = k_spectrum - k_evaluated_model
        # populate the plot with everything
        lines = {'linestyle': '-'}
        plt.rc('lines', **lines)
        line_ax[0][2].plot(k_wave_array[100:1950],k_spectrum[100:1950]/1E-14,drawstyle='steps-mid')
        line_ax[0][2].plot(k_wave_array[100:1950],k_evaluated_model[100:1950]/1E-14,color='red',drawstyle='steps-mid',lw=2)
        line_ax[1][2].plot(k_wave_array[100:1950],k_weights_new[100:1950]/1E-14,color='green',drawstyle='steps-mid',lw=2)
        line_ax[0][2].plot(k_wave_array[100:1950],k_evaluated_model[100:1950]/1E-14,color='red')
        line_ax[0][2].fill_between(k_wave_array[100:1950],
                                   np.repeat(0,len(k_wave_array[100:1950])),
                                   k_evaluated_model[100:1950]/1E-14,
                                    facecolor='red',
                                    edgecolor='red',
                                    alpha=0.5)
        lines = {'linestyle': 'None'}
        plt.rc('lines', **lines)
        line_ax[2][2].errorbar(k_wave_array[100:1950],
                               k_residuals[100:1950]/1E-14,
                               ecolor='midnightblue',
                               yerr=[k_weights_new[100:1950]/1E-14,k_weights_new[100:1950]/1E-14],
                               marker='o',
                               markersize=5,
                               markerfacecolor='midnightblue',
                               markeredgecolor='midnightblue',
                               markeredgewidth=1,
                               capsize=1,
                               elinewidth=1)
        lines = {'linestyle': '--'}
        plt.rc('lines', **lines)
        line_ax[2][2].axhline(y=0,color='black')
        lines = {'linestyle': '-'}
        plt.rc('lines', **lines)

        # find out the y upper and lower limits
        y_limit = np.nanmax([(ha_flux/ha_model_sigma)/1E-14,(oiii5_flux/oiii_model_sigma)/1E-14])

        line_ax[0][2].set_xlim(k_wave_array[k_min_index],k_wave_array[k_max_index])
        line_ax[1][2].set_xlim(k_wave_array[k_min_index],k_wave_array[k_max_index])
        line_ax[2][2].set_xlim(k_wave_array[k_min_index],k_wave_array[k_max_index])
        line_ax[0][2].set_ylim(-0.1*y_limit,0.5*y_limit)
        line_ax[0][1].set_ylim(-0.1*y_limit,0.5*y_limit)
        line_ax[0][0].set_ylim(-0.1*y_limit,0.5*y_limit)
        line_ax[2][2].set_ylim(-5,5)
        line_ax[1][2].set_ylim((np.nanmin(k_weights_new[k_min_index:k_max_index]/1E-14)),
                               (np.nanmax(k_weights_new[k_min_index:k_max_index]/1E-14)))
        for ranges in k_sky_dict.values():
            line_ax[0][2].axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')
            line_ax[2][2].axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')
        yticks = line_ax[0][2].yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)
        # tick parameters 
        line_ax[0][2].tick_params(axis='both',
                       which='major',
                       labelsize=26,
                       length=12,
                       width=4)
        line_ax[0][2].tick_params(axis='both',
                       which='minor',
                       labelsize=26,
                       length=6,
                       width=4)
        line_ax[0][2].minorticks_on()
        line_ax[0][2].yaxis.get_offset_text().set_fontsize(26)
        [i.set_linewidth(4.0) for i in line_ax[0][2].spines.itervalues()]
        # tick parameters 
        line_ax[1][2].tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=12,
                       width=4)
        line_ax[1][2].tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=6,
                       width=4)
        [i.set_linewidth(4.0) for i in line_ax[1][2].spines.itervalues()]
        line_ax[2][2].tick_params(axis='both',
                       which='major',
                       labelsize=18,
                       length=12,
                       width=4)
        line_ax[2][2].tick_params(axis='both',
                       which='minor',
                       labelsize=18,
                       length=6,
                       width=4)
        [i.set_linewidth(4.0) for i in line_ax[2][2].spines.itervalues()]

        # create some labels for the wavebands
        
        line_ax[0][0].text(yj_wave_array[yj_min_index]+0.002,
                           0.45*y_limit,
                           r'$\textbf{YJ-Band}$',
                           fontsize=25,
                           fontweight='bold')
        line_ax[0][1].text(h_wave_array[h_min_index]+0.015,
                           0.45*y_limit,
                           r'$\textbf{H-Band}$',
                           fontsize=25,
                           fontweight='bold')
        line_ax[0][2].text(k_wave_array[k_max_index]-0.01,
                           0.45*y_limit,
                           r'$\textbf{K-Band}$',
                           fontsize=25,
                           fontweight='bold')

        # identify where the emission lines are

        lines = {'linestyle': '--'}
        plt.rc('lines', **lines)

        line_ax[0][0].axvline(x=oii_l_centre,color='black',lw=2)
        line_ax[0][0].text(oii_l_centre-0.0045,
                           0.35*y_limit,
                           r'$\textbf{[OII]$\lambda$3727}$',
                           fontsize=15,
                           fontweight='bold')
        line_ax[0][0].axvline(x=oii_h_centre,color='black',lw=2)
        line_ax[0][0].text(oii_h_centre+0.0005,
                           0.35*y_limit,
                           r'$\textbf{[OII]$\lambda$3729}$',
                           fontsize=15,
                           fontweight='bold')
        line_ax[0][1].axvline(x=hb_centre,color='black',lw=2)
        line_ax[0][1].text(hb_centre+0.0015,
                           0.35*y_limit,
                           r'$\textbf{H$\beta$}$',
                           fontsize=15,
                           fontweight='bold')
        line_ax[0][1].axvline(x=oiii4_centre,color='black',lw=2)
        line_ax[0][1].text(oiii4_centre-0.009,
                           0.35*y_limit,
                           r'$\textbf{[OIII]$\lambda$4960}$',
                           fontsize=15,
                           fontweight='bold')
        line_ax[0][1].axvline(x=oiii5_centre,color='black',lw=2)
        line_ax[0][1].text(oiii5_centre-0.009,
                           0.35*y_limit,
                           r'$\textbf{[OIII]$\lambda$5007}$',
                           fontsize=15,
                           fontweight='bold')
        line_ax[0][2].axvline(x=ha_centre,color='black',lw=2)
        line_ax[0][2].text(ha_centre-0.002,
                           0.35*y_limit,
                           r'$\textbf{H$\alpha$}$',
                           fontsize=15,
                           fontweight='bold')
        line_ax[0][2].axvline(x=nii_centre,color='black',lw=2)
        line_ax[0][2].text(nii_centre+0.0005,
                           0.35*y_limit,
                           r'$\textbf{[NII]}$',
                           fontsize=15,
                           fontweight='bold')

        # turn off the tick labels in some places
        line_ax[0][1].tick_params(labelleft='off')
        line_ax[0][2].tick_params(labelleft='off')
        line_ax[1][0].tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
        line_ax[1][1].tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
        line_ax[1][2].tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
        line_ax[2][1].tick_params(labelleft='off')
        line_ax[2][2].tick_params(labelleft='off')

        line_ax[2][1].set_xlabel(r'$\textbf{Wavelength[$\mu$m]}$',
                                 fontsize=30,
                                 fontweight='bold',
                                 labelpad=10)

        line_ax[0][0].set_ylabel(r'$\textbf{F$_{\lambda}$[10$^{-14}$ergs$^{-1}$cm$^{-2}$$\mu$m$^{-1}$]}$',
                            fontsize=25,
                            fontweight='bold',
                            labelpad=10)

#        line_ax[1][0].set_ylabel(r'$\textbf{Error spectrum}$',
#                            fontsize=20,
#                            fontweight='bold',
#                            labelpad=10)

        line_ax[2][0].set_ylabel(r'$\textbf{Fit residuals[10$^{-14}$ergs$^{-1}$cm$^{-2}$$\mu$m$^{-1}$]}$',
                            fontsize=15,
                            fontweight='bold',
                            labelpad=20)

        # final pieces to remove space between axes
        # tight layout and save
        line_fig.subplots_adjust(hspace=.0)
        #line_fig.subplots_adjust(vspace=.0)
        line_fig.tight_layout()
        #plt.show()
        line_fig.savefig(all_plot_save_name)
        plt.close('all')

    # examine how for this galaxy the line ratios fit into the BPT plane

    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('font', weight='bold')
    rc('text', usetex=True)
    rc('axes', linewidth=2)
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    # KBSS-MOSFIRE measurements
    kbss_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/KBSS_MOSFIRE_LINE_RATIOS/steidel14_line_ratios.txt')
    kbss_oiii_hb = kbss_table['log([OIII]/Hb)']
    kbss_oiii_hb_lower_error = kbss_table['e_log([OIII]/Hb)']
    kbss_oiii_hb_upper_error = kbss_table['E_log([OIII]/Hb)']
    kbss_nii_ha = kbss_table['log([NII]/Ha)']
    kbss_nii_ha_lower_error = kbss_table['e_log([NII]/Ha)']
    kbss_nii_ha_upper_error = kbss_table['E_log([NII]/Ha)']


    # sdss flux measurements
    sdss_table = fits.open('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/SDSS_LINE_FLUXES/sdss_reliable_flux_measurements.fits')
    sdss_oiii_hb = sdss_table[1].data['log_oiii_hb']
    sdss_nii_ha = sdss_table[1].data['log_nii_ha']

    fig, ax = plt.subplots(1,1, figsize=(10,10))

    ax.tick_params(axis='both',
                   which='major',
                   labelsize=26,
                   length=12,
                   width=4)
    ax.tick_params(axis='both', 
                   which='minor',
                   labelsize=26,
                   length=6,
                   width=4)

    ax.set_xlabel(r'log([NII] $\lambda$6585/H$\alpha$)',
                  fontsize=30,
                  fontweight='bold',
                  labelpad=15)

    ax.set_ylabel(r'log([OIII] $\lambda$5008/H$\beta$)',
                  fontsize=30,
                  fontweight='bold',
                  labelpad=15)

    lines = {'linestyle': 'None'}
    plt.rc('lines', **lines)

    cm = plt.cm.get_cmap('Blues')

    # try creating contour plot instead

    xedges = np.arange(-1.6,0.6,0.01)
    yedges = np.arange(-1.1,1.1,0.01)

    H, xedges, yedges = np.histogram2d(sdss_nii_ha,sdss_oiii_hb,bins=[xedges,yedges])

    diff_x = xedges[1] - xedges[0]
    diff_y = yedges[1] - yedges[0]

    xedges = xedges[:-1] + (diff_x / 2.0)
    yedges = yedges[:-1] + (diff_y / 2.0)

    X,Y = np.meshgrid(xedges,yedges)

    CS_blue = ax.contour(X,Y,H.T,
                         levels=np.arange(5,150,5.0),
                         cmap=cm,
                         label='SDSS z < 0.2 reliable Brinchmann+06')

    [i.set_linewidth(4.0) for i in ax.spines.itervalues()]

    for entry in CS_blue.collections:
        plt.setp(entry,linewidth=2.75)

    # plotting the KBSS redshift 2 points
    kbss_errorbar = ax.errorbar(kbss_nii_ha,
                                kbss_oiii_hb,
                                ecolor='black',
                                xerr=[kbss_nii_ha_lower_error,kbss_nii_ha_upper_error],
                                yerr=[kbss_oiii_hb_lower_error,kbss_oiii_hb_upper_error],
                                marker='o',
                                markersize=3,
                                markerfacecolor='green',
                                markeredgecolor='green',
                                markeredgewidth=2,
                                capsize=2,
                                elinewidth=2,
                                alpha=0.75,
                                label='KBSS-MOSFIRE: Steidel+14')
    # plotting the KLP GALAXY region points
    klps_errorbar = ax.errorbar(klp_log_nii_ha,
                                klp_log_oiii_hb,
                                ecolor='black',
                                xerr=[klp_lower_error_log_nii_ha,klp_upper_error_log_nii_ha],
                                yerr=[klp_lower_error_log_oiii_hb,klp_upper_error_log_oiii_hb],
                                marker='*',
                                markersize=20,
                                markerfacecolor='red',
                                markeredgecolor='red',
                                markeredgewidth=2,
                                capsize=3,
                                elinewidth=3,
                                label='KLEVER')
    ax.legend(loc='best',
                prop={'size':18,'weight':'bold'},
                frameon=False,
                markerscale=1.2,
                numpoints=1)
    ax.set_xlim(-1.7,0.0)
    ax.set_ylim(-0.8,1.1)
    fig.tight_layout()
    #plt.show()
    fig.savefig(plot_save_dir + 'bpt.png')
    plt.close('all')

    # return the emission line strengths, errors, sigma values
    # and errors for outputting to a table

    return [[oii_total_flux,oii_total_flux_lower_error,oii_total_flux_upper_error,oii_total_sn,oii_total_sn_lower_error,oii_total_sn_upper_error,oii_sigma,oii_sigma_lower_error,oii_sigma_upper_error],
            [hb_flux,hb_flux_lower_error,hb_flux_upper_error,hb_sn,hb_sn_lower_error,hb_sn_upper_error],
            [oiii5_flux,oiii5_flux_lower_error,oiii5_flux_upper_error,oiii5_sn,oiii5_sn_lower_error,oiii5_sn_upper_error,oiii5_sigma,oiii5_sigma_lower_error,oiii5_sigma_upper_error],
            [ha_flux,ha_flux_lower_error,ha_flux_upper_error,ha_sn,ha_sn_lower_error,ha_sn_upper_error,ha_sigma,ha_sigma_lower_error,ha_sigma_upper_error],
            [nii_flux,nii_flux_lower_error,nii_flux_upper_error,nii_sn,nii_sn_lower_error,nii_sn_upper_error],
            [oiii_hb_ratio,lower_error_oiii_hb_ratio,upper_error_oiii_hb_ratio,log_oiii_hb,lower_error_log_oiii_hb,upper_error_log_oiii_hb],
            [nii_ha_ratio,lower_error_nii_ha_ratio,upper_error_nii_ha_ratio,log_nii_ha,lower_error_log_nii_ha,upper_error_log_nii_ha],
            [oiii_oii_ratio,oiii_oii_ratio_lower_error,oiii_oii_ratio_upper_error,log_oiii_oii,log_oiii_oii_lower_error,log_oiii_oii_upper_error],
            [r23,r23_lower_error,r23_upper_error,log_r23,log_r23_lower_error,log_r23_upper_error]]

def perturb_value(noise,
                  flux_array):

    """
    Def:
    Take the flux array and perturb each component of that by the
    corresponding component in the noise array and return a new array
    of the same dimensions as the original flux_array. Useful for Monte
    Carlo and checking whether gaussian fitting is accurate enough

    Input:

            noise - single value for the noise
            flux_array - array containing the flux values

    Output:
            new_flux - containing the perturbed values

    """
    # check for zero values in the noise array
    noise[noise==0]=1E-25

    # construct the new flux array
    ran_array = np.random.normal(scale=abs(noise), size=len(flux_array))

    # do the perturbation using a gaussian distributed value
    # with mean of the flux array and sigma of the noise value

    return ran_array + flux_array

all_line_flux_file = '/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/DS9_REGIONS/klever_integrated_line_fluxes.txt'

if os.path.isfile(all_line_flux_file):

    os.system('rm %s' % all_line_flux_file)

with open(all_line_flux_file, 'a') as f:

    column_names = ['NAME',
                    'OII_FLUX',
                    'OII_FLUX_LOWER_ERROR',
                    'OII_FLUX_UPPER_ERROR',
                    'OII_SN',
                    'OII_SN_LOWER_ERROR',
                    'OII_SN_UPPER_ERROR',
                    'OII_SIGMA',
                    'OII_SIGMA_LOWER_ERROR',
                    'OII_SIGMA_UPPER_ERROR',
                    'HB_FLUX',
                    'HB_FLUX_LOWER_ERROR',
                    'HB_FLUX_UPPER_ERROR',
                    'HB_SN',
                    'HB_SN_LOWER_ERROR',
                    'HB_SN_UPPER_ERROR',
                    'OIII_FLUX',
                    'OIII_FLUX_LOWER_ERROR',
                    'OIII_FLUX_UPPER_ERROR',
                    'OIII_SN',
                    'OIII_SN_LOWER_ERROR',
                    'OIII_SN_UPPER_ERROR',
                    'OIII_SIGMA',
                    'OIII_SIGMA_LOWER_ERROR',
                    'OIII_SIGMA_UPPER_ERROR',
                    'HA_FLUX',
                    'HA_FLUX_LOWER_ERROR',
                    'HA_FLUX_UPPER_ERROR',
                    'HA_SN',
                    'HA_SN_LOWER_ERROR',
                    'HA_SN_UPPER_ERROR',
                    'HA_SIGMA',
                    'HA_SIGMA_LOWER_ERROR',
                    'HA_SIGMA_UPPER_ERROR',
                    'NII_FLUX',
                    'NII_FLUX_LOWER_ERROR',
                    'NII_FLUX_UPPER_ERROR',
                    'NII_SN',
                    'NII_SN_LOWER_ERROR',
                    'NII_SN_UPPER_ERROR',
                    'OIII/HB',
                    'OIII/HB_LOWER_ERROR',
                    'OIII/HB_UPPER_ERROR',
                    'LOG_OIII/HB',
                    'LOG_OIII/HB_LOWER_ERROR',
                    'LOG_OIII/HB_UPPER_ERROR',
                    'NII/HA',
                    'NII/HA_LOWER_ERROR',
                    'NII/HA_UPPER_ERROR',
                    'LOG_NII/HA',
                    'LOG_NII/HA_LOWER_ERROR',
                    'LOG_NII/HA_UPPER_ERROR',
                    'OIII/OII',
                    'OIII/OII_LOWER_ERROR',
                    'OIII/OII_UPPER_ERROR',
                    'LOG_OIII/OII',
                    'LOG_OIII/OII_LOWER_ERROR',
                    'LOG_OIII/OII_UPPER_ERROR',
                    'R23',
                    'R23_LOWER_ERROR',
                    'R23_UPPER_ERROR',
                    'LOG_R23',
                    'LOG_R23_LOWER_ERROR',
                    'LOG_R23_UPPER_ERROR']

    for item in column_names:

        f.write('%s\t' % item)

    f.write('\n')

    k_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Kband/KLP_K_NAMES_FINAL.txt')
    k_names = k_table['Filename']
    redshifts = k_table['redshift']
    h_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Hband/KLP_H_NAMES_FINAL.txt')
    h_names = h_table['Filename']
    yj_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES_FINAL.txt')
    yj_names = yj_table['Filename']
    region_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/DS9_REGIONS/regions_list_integrated.txt')
    region_names = region_table['Filename']
    plot_dirs = region_table['PLOT_DIR']

    for k_name, h_name, yj_name, redshift, reg, plot_dir in zip(k_names,
                                                                h_names,
                                                                yj_names,
                                                                redshifts,
                                                                region_names,
                                                                plot_dirs):
        

        all_line_fluxes = fit_all_regions(reg,
                                          yj_name,
                                          h_name,
                                          k_name,
                                          redshift,
                                          plot_dir,
                                          'klp',
                                          weight=True,
                                          spatial_smooth=False,
                                          smoothing_psf=4,
                                          spectral_smooth=False,
                                          spectral_smooth_width=2,
                                          make_plot=False,
                                          mcmc_output=True,
                                          mcmc_cycles=250)

        gal_name = k_name[len(k_name) - k_name[::-1].find("/"):]
        gal_name = gal_name[26:-5]

        f.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (gal_name,
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][6],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][7],
                                                                                                                                                                                                                                                                                      all_line_fluxes[0][8],
                                                                                                                                                                                                                                                                                      all_line_fluxes[1][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[1][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[1][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[1][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[1][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[1][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][6],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][7],
                                                                                                                                                                                                                                                                                      all_line_fluxes[2][8],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][6],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][7],
                                                                                                                                                                                                                                                                                      all_line_fluxes[3][8],
                                                                                                                                                                                                                                                                                      all_line_fluxes[4][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[4][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[4][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[4][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[4][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[4][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[5][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[5][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[5][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[5][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[5][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[5][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[6][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[6][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[6][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[6][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[6][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[6][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[7][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[7][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[7][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[7][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[7][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[7][5],
                                                                                                                                                                                                                                                                                      all_line_fluxes[8][0],
                                                                                                                                                                                                                                                                                      all_line_fluxes[8][1],
                                                                                                                                                                                                                                                                                      all_line_fluxes[8][2],
                                                                                                                                                                                                                                                                                      all_line_fluxes[8][3],
                                                                                                                                                                                                                                                                                      all_line_fluxes[8][4],
                                                                                                                                                                                                                                                                                      all_line_fluxes[8][5]))
        f.flush()
f.close()

