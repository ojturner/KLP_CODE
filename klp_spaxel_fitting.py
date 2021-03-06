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
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import poly1d
from sys import stdout
from photutils import CircularAperture
from photutils import EllipticalAperture
from photutils import aperture_photometry

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

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

def multi_vel_field_stott(infile,
                          line,
                          threshold,
                          g_c_min,
                          g_c_max,
                          ntimes=200,
                          spatial_smooth=True,
                          spectral_smooth=False,
                          smoothing_psf=0.2,
                          spectral_smooth_width=2,
                          prog='klp',
                          emp_mask=True,
                          weight_fit=False,
                          **kwargs):

    """
    Def: Use the stott_velocity_field method from the cube_class
    to create postage stamp images of the flux, velocity and dispersion
    including marks on the velocity image to show the flux centre.

    Input:
            infile - file containing the object name and the centre
                        coordinates
            line - emission line to fit
            threshold - s/n threshold to exceed
            **kwargs
            tol - (default of 40)
            method - either sum, median or mean. This determines how the
                        spaxels are combined if stacking is necessary

    """
    # read in the table of cube names
    Table = ascii.read(infile)

    # assign variables to the different items in the infile
    for entry in Table:

        obj_name = entry['Filename']

        cube = cubeOps(obj_name)

        redshift = float(entry['redshift'])

        sky_cube = entry['sky_cube']

        mask_x_lower = entry['mask_x_lower']

        mask_x_upper = entry['mask_x_upper']

        mask_y_lower = entry['mask_y_lower']

        mask_y_upper = entry['mask_y_upper']

        noise_method = 'mask'

        # define the science directory for each cube
        sci_dir = obj_name[:len(obj_name) - obj_name[::-1].find("/") - 1]

        print "\nDoing %s (redshift = %.3f) ..." % (obj_name, redshift)

        try:

            if kwargs['tol']:

                tolerance = kwargs['tol']

            else:

                tolerance = 30

        except KeyError:

            tolerance = 30

        try:

            if kwargs['method']:

                stack_method = kwargs['method']

            else:

                stack_method = 'median'

        except KeyError:

            stack_method = 'median'

        vel_field_stott_binning(obj_name,
                                sky_cube,
                                line,
                                redshift,
                                threshold,
                                mask_x_lower,
                                mask_x_upper,
                                mask_y_lower,
                                mask_y_upper,
                                g_c_min,
                                g_c_max,
                                tol=tolerance,
                                method=stack_method,
                                noise_method=noise_method,
                                ntimes=ntimes,
                                spatial_smooth=spatial_smooth,
                                spectral_smooth=spectral_smooth,
                                smoothing_psf=smoothing_psf,
                                spectral_smooth_width=spectral_smooth_width,
                                prog=prog,
                                emp_mask=emp_mask,
                                weight_fit=weight_fit)

def vel_field_stott_binning(incube,
                            sky_cube,
                            line,
                            redshift,
                            threshold,
                            mask_x_lower,
                            mask_x_upper,
                            mask_y_lower,
                            mask_y_upper,
                            g_c_min,
                            g_c_max,
                            tol=60,
                            method='median',
                            noise_method='cube',
                            ntimes=200,
                            spatial_smooth=False,
                            spectral_smooth=False,
                            smoothing_psf=0.3,
                            spectral_smooth_width=2,
                            prog='klp',
                            emp_mask=True,
                            weight_fit=False):

    """
    Def:
    Yes another method for computing the velocity field.
    This time using an optimised signal to noise computation.
    The noise will be computed by examining stacked cube pixels that
    don't contain the object.

    Input:
            line - emission line to fit, must be either oiii, oii, hb
            combine_file - file containing names of frames going into the
                cube stack
            redshift - the redshift value of the incube
            threshold - signal to noise threshold for the fit
            tol - error tolerance for gaussian fit (default of 40)
            method - stacking method when binning pixels
    Output:
            signal array, noise array - for the given datacube
    """
    # speed of light
    c = 2.99792458E5

    # check that the emission line choice is valid

    if not(line != 'oiii' or line != 'oii' or line != 'hb' or line != 'ha'):

        raise ValueError('Please ensure that you have'
                         + ' chosen an appropriate emission line')

    # open the data
    cube = cubeOps(incube)

    # get the wavelength array

    wave_array = cubeOps(incube).wave_array

    skycube = cubeOps(sky_cube)

    sky_wave = skycube.wave_array

    sky_data = skycube.data

    sky_x_dim = sky_data.shape[1]

    sky_y_dim = sky_data.shape[2]

    data = cube.data

    noise = cube.Table[2].data

    # add in the optional spatial smoothing component
    # to check whether this improves the cosmetics of the
    # line grids, although to be sure - all emission line fits
    # used for the line ratios are performed on the unsmoothed cube

    if spatial_smooth and spectral_smooth:

        print '[INFO:  ] smoothing data spatially with %s filter' % smoothing_psf
        print '[INFO:  ] smoothing data spectrally with %s filter' % spectral_smooth_width

        # first have to set nan values to 0

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if np.isnan(data[i,j,k]):
                        data[i,j,k] = 0.0
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                for k in range(noise.shape[2]):
                    if np.isnan(noise[i,j,k]):
                        noise[i,j,k] = 0.0

        sigma_g = (smoothing_psf / 0.1) / 2.355
        data = scifilt.gaussian_filter(data,
                                       sigma=[spectral_smooth_width,sigma_g,sigma_g])
        noise = scifilt.gaussian_filter(noise,
                                        sigma=[spectral_smooth_width,sigma_g,sigma_g])

        # the error spectrum depends sensitively on the smoothing
        # applied. If we want to smooth both spatially and spectrally
        # this reduces the noise/weights array massively
        # define the skyline weights array
        weights_file = incube[:-5] + '_error_spectrum' + str(smoothing_psf).replace(".", "") + str(spectral_smooth_width) + '.fits'

        print '[INFO: ] using weights file: %s' % weights_file

    elif spatial_smooth and not(spectral_smooth):

        print '[INFO:  ] smoothing data spatially with %s filter' % smoothing_psf

        # first have to set nan values to 0
        # record the positions of the nan values, so that
        # they can be reset to nan after the smoothing
        nan_list_data = []
        nan_list_noise = []

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if np.isnan(data[i,j,k]):
                        data[i,j,k] = 0.0
                        nan_list_data.append([i,j,k])

        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                for k in range(noise.shape[2]):
                    if np.isnan(noise[i,j,k]):
                        noise[i,j,k] = 0.0
                        nan_list_noise.append([i,j,k])

        sigma_g = (smoothing_psf / 0.1) / 2.355
        data = scifilt.gaussian_filter(data,
                                       sigma=[0.0,sigma_g,sigma_g])
        noise = scifilt.gaussian_filter(noise,
                                        sigma=[0.0,sigma_g,sigma_g])

        # reset the nan entries
        for entry in nan_list_data:
            data[entry[0],entry[1],entry[2]] = np.nan
        for entry in nan_list_noise:
            noise[entry[0],entry[1],entry[2]] = np.nan

        # define the weights file
        weights_file = incube[:-5] + '_error_spectrum' + str(smoothing_psf).replace(".", "") + '.fits'

        print '[INFO: ] using weights file: %s' % weights_file

    elif spectral_smooth and not(spatial_smooth):

        print '[INFO:  ] smoothing data spectrally with %s filter' % spectral_smooth_width

        # first have to set nan values to 0

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    if np.isnan(data[i,j,k]):
                        data[i,j,k] = 0.0
        for i in range(noise.shape[0]):
            for j in range(noise.shape[1]):
                for k in range(noise.shape[2]):
                    if np.isnan(noise[i,j,k]):
                        noise[i,j,k] = 0.0

        sigma_g = (smoothing_psf / 0.1) / 2.355
        data = scifilt.gaussian_filter(data,
                                       sigma=[spectral_smooth_width,0.0,0.0])
        noise = scifilt.gaussian_filter(noise,
                                        sigma=[spectral_smooth_width,0.0,0.0])

        # define the weights file
        weights_file = incube[:-5] + '_error_spectrum' + str(spectral_smooth_width) + '.fits'

        print '[INFO: ] using weights file: %s' % weights_file

    else:

        weights_file = incube[:-5] + '_error_spectrum.fits' 

        print '[INFO:  ] No data smoothing selected'

        print '[INFO: ] using weights file: %s' % weights_file

    weights_array = fits.open(weights_file)[0].data

    xpixs = data.shape[1]

    ypixs = data.shape[2]

    # construct the instrumental resolution cube

    if cube.filter == 'K' or cube.filter == 'HK':

        l_tup = ([2.022,2.032],
                 [2.03,2.039],
                 [2.036,2.046],
                 [2.068,2.078],
                 [2.19,2.2])

    elif cube.filter == 'H':

        l_tup = ([1.521,1.527],
                 [1.600,1.607],
                 [1.687,1.693],
                 [1.698,1.704],
                 [1.710,1.716])

    else:

        l_tup = ([1.037,1.038],
                 [1.097,1.098],
                 [1.153,1.155],
                 [1.228,1.2295],
                 [1.301,1.303])

    res_array = []

    for entry in l_tup:

        res_array.append(sky_res(sky_data,
                                 sky_wave,
                                 sky_x_dim,
                                 sky_y_dim,
                                 entry[0],
                                 entry[1]))

    # now construct the 2D grid of values

    sky_res_grid = np.full((xpixs,ypixs),
                           np.mean(res_array))

    sky_res_error_grid = np.full((xpixs,ypixs),
                                 np.std(res_array))

    # set the search limits in the different filters
    # this is to account for differing spectral resolutions

    if cube.filter == 'K':

        # limits for the search for line-peak

        lower_t = 8
        upper_t = 9

        # limits for the search for signal computation
        # and gaussian fitting the emission line

        range_lower = 9
        range_upper = 10

    elif cube.filter == 'HK':

        lower_t = 5
        upper_t = 6

        range_lower = 5
        range_upper = 6

    elif cube.filter == 'YJ':

        lower_t = 8
        upper_t = 9

        range_lower = 12
        range_upper = 13

    else:

        lower_t = 8
        upper_t = 9

        range_lower = 9
        range_upper = 10

#    # find the polynomial to subtract from each spaxel (thermal noise)
#    if cube.filter == 'K' or cube.filter == 'HK':
#        poly_best = noise_from_mask_poly_subtract(cube.filter,
#                                                  data,
#                                                  mask_x_lower,
#                                                  mask_x_upper,
#                                                  mask_y_lower,
#                                                  mask_y_upper)
#        # update the data to have this thermal noise subtracted
#        for i in range(xpixs):
#            for j in range(ypixs):
#                data[:, i, j] = data[:, i, j] - poly_best
#    # now thermal noise subtracted and everything can proceed
#    # as before - or alternatively can get rid of this step again

    if line == 'oiii':

        central_wl = 0.500824 * (1. + redshift)

    elif line == 'oiiiweak':

        central_wl = 0.4960295 * (1. + redshift)

    elif line == 'hb':

        central_wl = 0.486268 * (1. + redshift)

    elif line == 'oii':

        central_wl = 0.3727485 * (1. + redshift)

    elif line == 'ha':

        central_wl = 0.6564614 * (1. + redshift)

    elif line == 'nii':

        central_wl = 0.658527 * (1. + redshift)

    # find the index of the chosen emission line
    line_idx = np.nanargmin(np.abs(wave_array - central_wl))

    # the shape of the data is (spectrum, xpixel, ypixel)
    # loop through each x and y pixel and get the OIII5007 S/N

    sn_array = np.empty(shape=(xpixs, ypixs))

    signal_array = np.empty(shape=(xpixs, ypixs))

    noise_array = np.empty(shape=(xpixs, ypixs))

    vel_array = np.empty(shape=(xpixs, ypixs))

    disp_array = np.empty(shape=(xpixs, ypixs))

    flux_array = np.empty(shape=(xpixs, ypixs))

    flux_error_array = np.empty(shape=(xpixs, ypixs))

    vel_error_array = np.empty(shape=(xpixs, ypixs))

    sig_error_array = np.empty(shape=(xpixs, ypixs))

    # array to check the coincidence of gauss fit flux and
    # flux recovered by the sum

    measurement_array = np.empty(shape=(xpixs, ypixs))

    # rejection array to help identify why a spaxel got rejected
    # 0 = nan signal
    # 1 = accepted signal without binning
    # 1.3 = accepted signal 3x3 binning
    # 1.5 = accepted signal 5x5 binning
    # 2 = S/N < 0
    # 3 = S/N < T && S/G && Err
    # 4 = S/N < T && S/G
    # 5 = S/N < T # most common until we sort out binning effect
    # 6 = S/G
    # 7 = Err
    # 8 = S/G && Err
    # 9 = S/N && Err
    # 10 = unknown
    # 11 = even more unknown

    rejection_array = np.empty(shape=(xpixs, ypixs))

    # empirically mask the skylines
    # want to do this to increase S/N and
    # get rid of really bad skyline regions

    if emp_mask:

        print '[INFO:] Masking skylines empirically'

        # figure out which spectral pixels should be masked
        # across the cube on the basis on the median noise spectrum
        indices = empirically_mask_skylines(data,
                                            mask_x_lower,
                                            mask_x_upper,
                                            mask_y_lower,
                                            mask_y_upper)

        # and mask these points identified as crap
        for i in range(0,data.shape[1]):
            for j in range(0,data.shape[2]):
                data[:,i,j][indices] = np.nan
                noise[:,i,j][indices] = np.nan

    #####################################################
    #####################################################
    ########MEDIAN NOISE AND REDUCTION FACTORS###########
    #####################################################
    #####################################################
    print '[INFO:] CALCULATING NOISE PROPERTIES'
    # define the ranges for this experiment
    if line_idx < 250:
        noise_ranges = [[line_idx - 12,line_idx + 13],
                        [line_idx + 13,line_idx + 38],
                        [line_idx + 38,line_idx + 63],
                        [line_idx + 63,line_idx + 88],
                        [line_idx + 88,line_idx + 103],
                        [line_idx + 103,line_idx + 128],
                        [line_idx + 128,line_idx + 153]]

    elif line_idx > 1750:
        noise_ranges = [[line_idx - 153,line_idx - 128],
                        [line_idx - 128,line_idx - 103],
                        [line_idx - 103,line_idx - 88],
                        [line_idx - 88,line_idx - 63],
                        [line_idx - 63,line_idx - 38],
                        [line_idx - 38,line_idx - 13],
                        [line_idx - 13,line_idx + 12]]

    else:
        noise_ranges = [[line_idx - 88,line_idx - 63],
                        [line_idx - 63,line_idx - 38],
                        [line_idx - 38,line_idx - 13],
                        [line_idx - 103,line_idx + 12],
                        [line_idx + 12,line_idx + 37],
                        [line_idx + 37,line_idx + 62],
                        [line_idx + 62,line_idx + 87]]

    # compute the binned noise cubes
    three_binned_data = bin_three_for_noise(data)
    five_binned_data = bin_five_for_noise(data)

    # define the arrays to house the unbinned,3binned and 5binned noises
    unbinned_noise_list = []
    three_binned_noise_list = []
    five_binned_noise_list = []

    for entry in noise_ranges:
        unbinned_line_noise, unbinned_line_p_noise = noise_from_mask(data,
                                                                     entry[0],
                                                                     entry[1],
                                                                     mask_x_lower,
                                                                     mask_x_upper,
                                                                     mask_y_lower,
                                                                     mask_y_upper)
        unbinned_noise_list.append(unbinned_line_noise)
        three_binned_line_noise, three_binned_line_p_noise = noise_from_mask(three_binned_data,
                                                                             entry[0],
                                                                             entry[1],
                                                                             mask_x_lower,
                                                                             mask_x_upper,
                                                                             mask_y_lower,
                                                                             mask_y_upper)
        three_binned_noise_list.append(three_binned_line_noise)
        five_binned_line_noise, five_binned_line_p_noise = noise_from_mask(five_binned_data,
                                                                           entry[0],
                                                                           entry[1],
                                                                           mask_x_lower,
                                                                           mask_x_upper,
                                                                           mask_y_lower,
                                                                           mask_y_upper)
        five_binned_noise_list.append(five_binned_line_noise)

    # what's the median noise for this cube?
    print 'MEDIAN NOISE LEVEL: %s' % np.nanmedian(unbinned_noise_list)
    # now find tred and fred
    three_noise_ratio = np.array(unbinned_noise_list)/np.array(three_binned_noise_list)
    t_red = np.nanmax(three_noise_ratio)
    print 'MEDIAN THREE REDUCTION: %s' % (t_red)
    five_noise_ratio = np.array(unbinned_noise_list)/np.array(five_binned_noise_list)
    f_red = np.nanmax(five_noise_ratio)
    print 'MEDIAN FIVE REDUCTION: %s' % (f_red)

    for i in range(0, xpixs, 1):

        for j in range(0, ypixs, 1):

            stdout.write("\r %.1f%% complete" % (100 * float(i + 1) / xpixs))
            stdout.flush()

            # print 'Fitting Spaxel %s/%s %s/%s' % (i, xpixs - 1, j, ypixs - 1)
            # want to do all the gaussian fitting and calculations on
            # normalised data, since some values in the cubes are as small as
            # 1E-22 and this gives errors during the fitting procedure
            # divide both the spaxel spec and the noise by the median value
            # of the spaxel spec, and then convert back at the end.
            # The only thing this could change is the flux, neither the
            # velocity or the velocity dispersion should be affected by this

            spaxel_spec = data[:, i, j]
            spaxel_noise = noise[:, i, j]

            #################################
            #################################
            #EXPERIMENTAL NOISE CUBE MASKING#
            #        OF THE SKYLINES        #
            #################################
            #################################
            # first mask the noise cube spaxel at the 
            # locations of the known skylines
#            masked_spaxel_noise = mask_the_sky.masking_sky(wave_array,
#                                                           spaxel_noise,
#                                                           cube.filter)
#            # then create a new masked_noise_spec where the noise goes
#            # above a threshold of 3sigma of the standard deviation of
#            # this masked spaxel noise
#            new_masked_noise_spec = np.ma.masked_where(spaxel_noise > 3 * np.nanstd(masked_spaxel_noise),spaxel_noise)
#            # use the mask from this procedure and apply it to the data
#            mask = new_masked_noise_spec.mask
#            spaxel_spec_masked = np.ma.array(spaxel_spec,mask=mask)
#            spaxel_spec = spaxel_spec_masked.filled(fill_value=np.nan)

            # procede as normal after this and hopefully everything works?

            #print 'FILTER: %s' % cube.filter
            spaxel_median_flux = abs(np.nanmedian(spaxel_spec[350:1500]))
            noise_from_hist, noise_centre = noise_from_histogram(wave_array,
                                                                 spaxel_spec/spaxel_median_flux,
                                                                 redshift,
                                                                 cube.filter,
                                                                 prog)

            # just assume flat continuum for now
            spaxel_spec = spaxel_spec - (noise_centre*spaxel_median_flux)
            spaxel_median_flux = abs(np.nanmedian(spaxel_spec[350:1500]))

            # here subtract the continuum from the spaxel BEFORE
            # measuring the line counts. Only do this if the whole
            # array isn't nan

#            if not np.isnan(spaxel_median_flux):
#                cont = continuum_subtract_full(spaxel_spec,
#                                               wave_array,
#                                               redshift,
#                                               cube.filter)
#                if cube.filter == 'K':
#                    spaxel_spec[0:1600] = spaxel_spec[0:1600] - cont[0:1600]
#                elif cube.filter == 'H':
#                    spaxel_spec[0:1700] = spaxel_spec[0:1700] - cont[0:1700]
#                elif cube.filter == 'YJ':
#                    spaxel_spec[0:1900] = spaxel_spec[0:1900] - cont[0:1900]
#            # recompute the spaxel median flux for consistency
#            spaxel_median_flux = abs(np.nanmedian(spaxel_spec[350:1500]))            

            # add failsafe for the median flux being nan
            if np.isnan(spaxel_median_flux):
                spaxel_median_flux = 5E-19

            weights_array_norm = weights_array / spaxel_median_flux
            spaxel_spec = spaxel_spec/spaxel_median_flux
            spaxel_noise = noise[:, i, j]/spaxel_median_flux
            # first search for the linepeak, which may be different
            # to that specified by the systemic redshift
            # set the upper and lower ranges for the t_index search
            # need this try/except statement just in case everything
            # is nan

            try:
                t_index = np.nanargmax(spaxel_spec[line_idx - lower_t:
                                                   line_idx + upper_t])
            except ValueError:
                t_index = 10

            # need this to be an absolute index
            t_index = t_index + line_idx - lower_t

            # then sum the flux inside the region over which the line
            # will be. Width of line is roughly 0.003, which is 10
            # spectral elements in K and 6 in HK

            lower_limit = t_index - range_lower
            upper_limit = t_index + range_upper

            # take the line counts as the sum over the relevant lambda range
            # but only want to keep the positive values

            positive_counts_index = np.where(spaxel_spec[lower_limit:upper_limit] > 0)

            line_counts = np.nansum(spaxel_spec[lower_limit:
                                                upper_limit][positive_counts_index])

#            noise_from_masked_method = noise_from_masked_spectrum(wave_array,
#                                                                  spaxel_spec,
#                                                                  redshift,
#                                                                  len(spaxel_spec[lower_limit:upper_limit]),
#                                                                  line_idx,
#                                                                  cube.filter)

            # define the reduced weights arrays
            unbinned_weights_array_norm = weights_array_norm[lower_limit: upper_limit]
            t_red_weights_array_norm = weights_array_norm[lower_limit: upper_limit]/t_red
            f_red_weights_array_norm = weights_array_norm[lower_limit: upper_limit]/f_red

            # and the fitting wavelength
            fit_wave_array = wave_array[lower_limit: upper_limit]

            # and the fit spaxel spec
            fit_spectrum_array = spaxel_spec[lower_limit: upper_limit]

            # multiply the line counts by the cube wavelength separation
            line_counts = line_counts * cube.dL

            # do the gaussian fitting

            plt.close('all')

            # fit the oii line with a double gaussian if this
            # is what is being considered

            if line == 'oii':

                try:

                    gauss_values, covar = oii_gauss_fit(fit_wave_array,
                                                        fit_spectrum_array,
                                                        redshift,
                                                        unbinned_weights_array_norm,
                                                        central_wl,
                                                        weight_fit)

                except TypeError:

                    gauss_values = {'amplitude': np.nan,
                                    'sigma': np.nan,
                                    'center': np.nan}

                    covar = np.zeros(shape=(3, 3))

                    covar = covar * np.nan

            else:

                try:

                    gauss_values, covar = gauss_fit(fit_wave_array,
                                                    fit_spectrum_array,
                                                    unbinned_weights_array_norm,
                                                    central_wl,
                                                    weight_fit)

                except TypeError:

                    gauss_values = {'amplitude': np.nan,
                                    'sigma': np.nan,
                                    'center': np.nan}

                    covar = np.zeros(shape=(3, 3))

                    covar = covar * np.nan


            # define the ratio of line counts to the gaussian fitting flux

            int_ratio = line_counts / gauss_values['amplitude']

            # assign variables to the gaussian fitting errors
            # sometimes if fitting is so poor the errors are not defined
            # define the errors as infinite in this case

            try:

                amp_err = 100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']

                sig_err = 100 * np.sqrt(covar[0][0]) / gauss_values['sigma']

                cen_err = 100 * np.sqrt(covar[1][1]) / gauss_values['center']

            # if the error is thrown assign infinite errors
            except TypeError:

                amp_err = np.inf

                sig_err = np.inf

                cen_err = np.inf

            # set up a 2D array to examine the variation in the
            # difference between trapezoidal and gaussian errors

            try:

                measurement_array[i][j] = int_ratio

            except TypeError:

                measurement_array[i][j] = np.nan

            # compute the line noise using the mask technique
            # this is the only difference between the mask noise
            # and vel field sigma methods - change this so that there
            # is an if statement here to distinguish between the two
            # methods
            if noise_method == 'mask':

                line_noise, line_p_noise = noise_from_mask(data,
                                                           lower_limit,
                                                           upper_limit,
                                                           mask_x_lower,
                                                           mask_x_upper,
                                                           mask_y_lower,
                                                           mask_y_upper)

                line_noise = line_noise/spaxel_median_flux
                line_p_noise = line_p_noise/spaxel_median_flux

            elif noise_method == 'cube':

                sigma_array = spaxel_noise[lower_limit:upper_limit]

                sigma_squared = sigma_array * sigma_array

                line_noise = np.sqrt(np.nansum(sigma_squared))

                line_p_noise = np.std(sigma_array)

            else:

                print 'Please Provide valid noise method'

                raise ValueError('Please provide valid noise method')

            # print 'NOISE COMPARISON %s %s' % (line_noise*spaxel_median_flux,line_p_noise*spaxel_median_flux)

            # find the noise reduction factors of the binning methods
            # these feed into the binning_three and binning_five
            # methods to figure out what the new noise should be

            # get the three binned and five binned cubes for
            # new noise calculation for this spaxel
#            t_red = compute_noise_reduction_factor_three(data,
#                                                         xpixs,
#                                                         ypixs,
#                                                         lower_limit,
#                                                         upper_limit)
#            print '3 noise reduction factor: %s' % (t_red)
#            f_red = compute_noise_reduction_factor_five(data,
#                                                        xpixs,
#                                                        ypixs,
#                                                        lower_limit,
#                                                        upper_limit)
#            print '5 noise reduction factor: %s' % (f_red)

            #print 'REDUCTION FACTORS %s %s' % (t_red,f_red)

            # this must also be multiplied by the spectral resolution
            # print 'This is the original line noise: %s' % line_p_noise

            # multiply line noise by the wavelength separation
            line_noise = line_noise * cube.dL
#            noise_from_masked_method = noise_from_masked_method * cube.dL

#                print 'THIS IS THE SIGNAL %s' % line_counts
#                print 'THIS IS THE NOISE %s' % line_noise

            # be careful with how the signal array is populated

            if np.isnan(line_counts):

                signal_array[i, j] = 0

            else:

                signal_array[i, j] = line_counts*spaxel_median_flux

            # print 'LINE COUNTS CHECK: %s %s' % (line_counts,line_counts*spaxel_median_flux)

            # populate the noise array

            noise_array[i, j] = line_noise*spaxel_median_flux

            # compute the signal to noise on the basis of the
            # above calculations

            line_sn = line_counts / line_noise
            sn_array[i, j] = line_sn

            #print 'THIS IS THE SIGNAL TO NOISE: %s %s %s' % (line_sn,line_counts,line_noise)

            # searching the computed signal to noise in this section

            if np.isnan(line_sn) or np.isinf(line_sn) or np.isclose(line_sn, 0, atol=1E-5):

                # print 'getting rid of nan'

                # we've got a nan entry - get rid of it

                vel_array[i, j] = np.nan
                disp_array[i, j] = np.nan
                flux_array[i, j] = np.nan
                flux_error_array[i, j] = np.nan
                vel_error_array[i, j] = np.nan
                sig_error_array[i, j] = np.nan
                rejection_array[i, j] = 0

            # initial checks to see if gaussian should be fit
            # the first conditions are that the difference between
            # the trapezoidal and gaussian integrals is less than
            # 25 percent on either side and that the signal to noise
            # value is greater than 5

            # also have a set of constraints based on the gaussian
            # fitting uncertainties - need these to be good (< 20%) to
            # proceed with the computation of the galaxy properties
            # otherwise don't just throw away - pass through for binning

            elif (line_sn > threshold) and \
                 (int_ratio >= g_c_min and int_ratio <= g_c_max) and \
                 (amp_err < tol and sig_err < tol and cen_err < tol):

                # print 'CRITERIA SATISFIED %s %s %s %s' % (i, j, line_sn, int_ratio)

                # plt.show()
                # do stuff - calculate the velocity

                # if the gaussian does not fit correctly this can throw
                # a nonetype error, since covar is empty

                # to get a handle on the parameter errors
                # going to run an MCMC each time a velocity point
                # is accepted

                mc_sig_array = []
                mc_amp_array = []
                mc_centre_array = []

                # print 'This is the noise: %s' % p_line_noise
                # print 'This is the signal: %s' % spaxel_spec

                v_o = c * (gauss_values['center'] - central_wl) / central_wl

                for loop in range(0, ntimes):

                    # print 'fitting %sth gaussian' % loop

                    # get the perturbed array using the helper function
                    new_flux = perturb_value(unbinned_weights_array_norm,
                                             fit_spectrum_array)

                    if line == 'oii':

                        # fit the gaussian to recover the parameters
                        gauss_values, covar = oii_gauss_fit(fit_wave_array,
                                                            new_flux,
                                                            redshift,
                                                            unbinned_weights_array_norm,
                                                            central_wl,
                                                            weight_fit)
                    else:
                        # fit the gaussian to recover the parameters
                        gauss_values, covar = gauss_fit(fit_wave_array,
                                                        new_flux,
                                                        unbinned_weights_array_norm,
                                                        central_wl,
                                                        weight_fit)

                    # plt.show()

                    # append the returned values to the mc arrays
                    # only if the errors are less than tol

                    try:

                        amp_err = 100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']

                        sig_err = 100 * np.sqrt(covar[0][0]) / gauss_values['sigma']

                        cen_err = 100 * np.sqrt(covar[1][1]) / gauss_values['center']

                    # if the error is thrown assign infinite errors
                    except TypeError:

                        amp_err = np.inf

                        sig_err = np.inf

                        cen_err = np.inf

                    if (amp_err < tol and sig_err < tol and cen_err < tol):

                        mc_sig_array.append(gauss_values['sigma'])
                        mc_amp_array.append(gauss_values['amplitude'])
                        mc_centre_array.append(gauss_values['center'])

                # print 'This is how many survived %s' % len(mc_sig_array)
                # np array the resultant mc arrays

                mc_sig_array = np.array(mc_sig_array)
                mc_amp_array = np.array(mc_amp_array)
                mc_centre_array = np.array(mc_centre_array)

                # make a histogram of the centre points and plot

#                    hist, edges = np.histogram(mc_centre_array)
#                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#                    ax.plot(edges[:-1], hist)
#                    plt.show()
#                    plt.close('all')

                c = 2.99792458E5

                # make histograms of the MCMC results to
                # determine the gaussian fitting parameters
                # from a gaussian fit to those

                vel_hist, vel_edges = np.histogram((c * ((mc_centre_array
                                         - central_wl) / central_wl)),
                                                   bins=ntimes / 20.0)

                sig_hist, sig_edges = np.histogram(c * (mc_sig_array / central_wl),
                                                   bins=ntimes / 20.0)

                amp_hist, amp_edges = np.histogram(mc_amp_array, bins=ntimes / 20.0)

                # make gaussian fits to these histograms

                vel_gauss_values, vel_covar = mc_gauss_fit(vel_edges[:-1],
                                                             vel_hist)

                sig_gauss_values, sig_covar = mc_gauss_fit(sig_edges[:-1],
                                                             sig_hist)

                amp_gauss_values, amp_covar = mc_gauss_fit(amp_edges[:-1],
                                                             amp_hist)

                rejection_array[i, j] = 1
                vel_array[i, j] = vel_gauss_values['center']

                # sometimes getting bung values for the width of
                # the emission lines

                if sig_gauss_values['center'] > 0 and sig_gauss_values['center'] < 3000:

                    disp_array[i, j] = sig_gauss_values['center']

                else:

                    disp_array[i, j] = np.nan

                flux_array[i, j] = amp_gauss_values['center']*spaxel_median_flux
                flux_error_array[i, j] = amp_gauss_values['sigma']*spaxel_median_flux
                vel_error_array[i, j] = vel_gauss_values['sigma']
                sig_error_array[i, j] = sig_gauss_values['sigma']

            # don't bother expanding area if line_sn starts negative

            elif line_sn < 0:

                # print 'Found negative signal %s %s' % (i, j)
                vel_array[i, j] = np.nan
                disp_array[i, j] = np.nan
                flux_array[i, j] = np.nan
                flux_error_array[i, j] = np.nan
                vel_error_array[i, j] = np.nan
                sig_error_array[i, j] = np.nan
                rejection_array = 2.0

            # If between 0 and the threshold, search surrounding area
            # for more signal - do this in the direction of the galaxy
            # centre (don't know if this introduces a bias to the
            # measurement or not)

            elif (line_sn > 0 and line_sn < threshold) or \
                 (line_sn > threshold and (int_ratio < g_c_min or int_ratio > g_c_max)) or \
                 (line_sn > threshold and (amp_err > tol or sig_err > tol or cen_err > tol)):

                # print 'Attempting to improve signal: %s %s %s' % (line_sn, i, j)

                # compute the stacked 3x3 spectrum using helper method

                spec, new_noise = binning_three(data,
                                                line_noise*spaxel_median_flux,
                                                i,
                                                j,
                                                lower_limit,
                                                upper_limit,
                                                t_red,
                                                method)

                spec = spec/spaxel_median_flux
                new_noise = new_noise/spaxel_median_flux
#                new_noise_from_histogram, new_noise_centre = noise_from_histogram(wave_array,
#                                                                                  spec,
#                                                                                  redshift,
#                                                                                  cube.filter)
#                new_noise_from_masked_method = noise_from_masked_spectrum(wave_array,
#                                                                  spec,
#                                                                  redshift,
#                                                                  len(spec[lower_limit:upper_limit]),
#                                                                  line_idx,
#                                                                  cube.filter)
#                new_noise_from_masked_method = new_noise_from_masked_method * cube.dL
                spec = spec[lower_limit:upper_limit]

                positive_counts_index = np.where(spec > 0)

                # now that spec has been computed, look at whether
                # the signal to noise of the stack has improved

                new_line_counts = np.nansum(spec[positive_counts_index])

                new_line_counts = new_line_counts * cube.dL

                new_sn = new_line_counts / new_noise

                # have to fit gaussian at this point as well
                # and examine similarity between the gaussian fit
                # and the line_counts
                plt.close('all')

                if line == 'oii':

                    gauss_values, covar = oii_gauss_fit(wave_array[lower_limit: upper_limit],
                                                        spec,
                                                        redshift,
                                                        t_red_weights_array_norm,
                                                        central_wl,
                                                        weight_fit)
                else:
                    gauss_values, covar = gauss_fit(wave_array[lower_limit: upper_limit],
                                                    spec,
                                                    t_red_weights_array_norm,
                                                    central_wl,
                                                    weight_fit)

                try:

                    amp_err = 100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']

                    sig_err = 100 * np.sqrt(covar[0][0]) / gauss_values['sigma']

                    cen_err = 100 * np.sqrt(covar[1][1]) / gauss_values['center']

                # if the error is thrown assign infinite errors
                except TypeError:

                    amp_err = np.inf

                    sig_err = np.inf

                    cen_err = np.inf

                int_ratio = new_line_counts / gauss_values['amplitude']

                # print 'did things improve: new %s old %s' % (new_sn, line_sn)

                # if the new signal to noise is greater than the
                # threshold, save this in the cube and proceed

                if (new_sn > threshold) and \
                   (int_ratio >= g_c_min and int_ratio <= g_c_max) and \
                   (amp_err < tol and sig_err < tol and cen_err < tol):

                    # print 'CRITERIA SATISFIED by 3x3 binning %s %s %s %s' % (i, j, new_sn, int_ratio)

                    # plt.show()
                    # do stuff - calculate the velocity

                    # if the gaussian does not fit correctly this can throw
                    # a nonetype error, since covar is empty

                    mc_sig_array = []
                    mc_amp_array = []
                    mc_centre_array = []
                    new_line_p_noise = line_p_noise / t_red

                    # print 'NEW NOISE COMPARISON: %s %s %s' % (new_noise,new_line_p_noise,new_noise_from_histogram)

                    # print 'This is the noise: %s' % new_line_p_noise
                    # print 'This is the signal: %s' % spec

                    for loop in range(0, ntimes):

                        # print 'fitting %sth gaussian' % loop

                        # get the perturbed array using the helper function
                        new_flux = perturb_value(t_red_weights_array_norm,
                                                 spec)

                        if line == 'oii':

                            # fit the gaussian to recover the parameters
                            gauss_values, covar = oii_gauss_fit(fit_wave_array,
                                                                new_flux,
                                                                redshift,
                                                                t_red_weights_array_norm,
                                                                central_wl,
                                                                weight_fit)
                        else:
                            # fit the gaussian to recover the parameters
                            gauss_values, covar = gauss_fit(fit_wave_array,
                                                            new_flux,
                                                            t_red_weights_array_norm,
                                                            central_wl,
                                                            weight_fit)

                        # plt.show()

                        # append the returned values to the mc arrays
                        # only if the errors are less than tol

                        try:

                            amp_err = 100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']

                            sig_err = 100 * np.sqrt(covar[0][0]) / gauss_values['sigma']

                            cen_err = 100 * np.sqrt(covar[1][1]) / gauss_values['center']

                        # if the error is thrown assign infinite errors
                        except TypeError:

                            amp_err = np.inf

                            sig_err = np.inf

                            cen_err = np.inf

                        if (amp_err < tol and sig_err < tol and cen_err < tol):

                            mc_sig_array.append(gauss_values['sigma'])
                            mc_amp_array.append(gauss_values['amplitude'])
                            mc_centre_array.append(gauss_values['center'])

                    # print 'This is how many survived %s' % len(mc_sig_array)
                    # np array the resultant mc arrays

                    mc_sig_array = np.array(mc_sig_array)
                    mc_amp_array = np.array(mc_amp_array)
                    mc_centre_array = np.array(mc_centre_array)

                    # np array the resultant mc arrays

                    mc_sig_array = np.array(mc_sig_array)
                    mc_amp_array = np.array(mc_amp_array)
                    mc_centre_array = np.array(mc_centre_array)

                    # make a histogram of the centre points and plot

#                    hist, edges = np.histogram(mc_centre_array)
#                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
#                    ax.plot(edges[:-1], hist)
#                    plt.show()
#                    plt.close('all')

                    c = 2.99792458E5

                    # make histograms of the MCMC results to
                    # determine the gaussian fitting parameters
                    # from a gaussian fit to those

                    vel_hist, vel_edges = np.histogram((c * ((mc_centre_array
                                             - central_wl) / central_wl)),
                                                       bins=ntimes / 20.0)

                    sig_hist, sig_edges = np.histogram(c * (mc_sig_array / central_wl),
                                                       bins=ntimes / 20.0)

                    amp_hist, amp_edges = np.histogram(mc_amp_array, bins=ntimes / 20.0)

                    # make gaussian fits to these histograms

                    vel_gauss_values, vel_covar = mc_gauss_fit(vel_edges[:-1],
                                                                 vel_hist)

                    sig_gauss_values, sig_covar = mc_gauss_fit(sig_edges[:-1],
                                                                 sig_hist)

                    amp_gauss_values, amp_covar = mc_gauss_fit(amp_edges[:-1],
                                                                 amp_hist)

                    # append the original line-sn rather than the binned sn
                    sn_array[i, j] = new_sn
                    rejection_array[i, j] = 1.3
                    vel_array[i, j] = vel_gauss_values['center']

                    if sig_gauss_values['center'] > 0 and sig_gauss_values['center'] < 3000:

                        disp_array[i, j] = sig_gauss_values['center']

                    else:

                        disp_array[i, j] = np.nan

                    flux_array[i, j] = amp_gauss_values['center']*spaxel_median_flux
                    flux_error_array[i, j] = amp_gauss_values['sigma']*spaxel_median_flux
                    vel_error_array[i, j] = vel_gauss_values['sigma']
                    sig_error_array[i, j] = sig_gauss_values['sigma']

                # don't bother expanding area if line_sn starts negative

                elif new_sn < 0:

                    # print 'Found negative signal %s %s' % (i, j)

                    vel_array[i, j] = np.nan
                    disp_array[i, j] = np.nan
                    flux_array[i, j] = np.nan
                    flux_error_array[i, j] = np.nan
                    vel_error_array[i, j] = np.nan
                    sig_error_array[i, j] = np.nan
                    rejection_array[i, j] = 2.0

                # If between 0 and the threshold, search surrounding area
                # for more signal - do this in the direction of the galaxy
                # centre (don't know if this introduces a bias to the
                # measurement or not)

                elif (new_sn > 0 and new_sn < threshold) or \
                     (new_sn > threshold and (int_ratio < g_c_min or int_ratio > g_c_max)) or \
                     (new_sn > threshold and (amp_err > tol or sig_err > tol or cen_err > tol)):

                    # try the 5x5 approach towards the cube centre

                    spec, final_noise = binning_five(data,
                                                     line_noise*spaxel_median_flux,
                                                     i,
                                                     j,
                                                     lower_limit,
                                                     upper_limit,
                                                     f_red,
                                                     method)

                    spec = spec / spaxel_median_flux
                    final_noise = final_noise / spaxel_median_flux
#                    final_noise_from_histogram, final_noise_centre = noise_from_histogram(wave_array,
#                                                                                          spec,
#                                                                                          redshift,
#                                                                                          cube.filter)
#                    final_noise_from_masked_method = noise_from_masked_spectrum(wave_array,
#                                                                  spec,
#                                                                  redshift,
#                                                                  len(spec[lower_limit:upper_limit]),
#                                                                  line_idx,
#                                                                  cube.filter)
#                    final_noise_from_masked_method = final_noise_from_masked_method * cube.dL
                    spec = spec[lower_limit:upper_limit]

                    positive_counts_index = np.where(spec > 0)  

                # now that spec has been computed, look at whether
                # the signal to noise of the stack has improved

                    final_line_counts = np.nansum(spec[positive_counts_index])

                    final_line_counts = cube.dL * final_line_counts

                    final_sn = final_line_counts / final_noise

                    plt.close('all')

                    if line == 'oii':

                        gauss_values, covar = oii_gauss_fit(fit_wave_array,
                                                            spec,
                                                            redshift,
                                                            f_red_weights_array_norm,
                                                            central_wl,
                                                            weight_fit)
                    else:

                        gauss_values, covar = gauss_fit(fit_wave_array,
                                                        spec,
                                                        f_red_weights_array_norm,
                                                        central_wl,
                                                        weight_fit)

                    try:

                        amp_err = 100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']

                        sig_err = 100 * np.sqrt(covar[0][0]) / gauss_values['sigma']

                        cen_err = 100 * np.sqrt(covar[1][1]) / gauss_values['center']

                    # if the error is thrown assign infinite errors
                    except TypeError:

                        amp_err = np.inf

                        sig_err = np.inf

                        cen_err = np.inf

                    int_ratio = final_line_counts / gauss_values['amplitude']

                    # print 'did things improve: final %s old %s' % (final_sn, new_sn)

                    # if the new signal to noise is greater than the
                    # threshold, save this in the cube and proceed

                    if (final_sn > threshold) and \
                       (int_ratio >= g_c_min and int_ratio <= g_c_max) and \
                       (amp_err < tol and sig_err < tol and cen_err < tol):

                        # time.sleep(2)

                        # add to the signal to noise array

                        # print 'CRITERIA SATISFIED AFTER 5x5 binning %s %s %s %s %s %s' % (i, j, final_sn, int_ratio, gauss_values['center'], gauss_values['sigma'])
                        # plt.show()

                        mc_sig_array = []
                        mc_amp_array = []
                        mc_centre_array = []
                        final_line_p_noise = line_p_noise / f_red

                        # print 'FINAL NOISE COMPARISON: %s %s %s' % (final_noise, final_line_p_noise, final_noise_from_histogram)

                        # print 'This is the final noise: %s' % final_line_p_noise
                        # print 'This is the signal: %s' % spec
                        # print f_red, line_p_noise, final_line_p_noise

                        for loop in range(0, ntimes):

                            # print 'fitting %sth gaussian' % loop

                            # get the perturbed array using the helper function
                            new_flux = perturb_value(f_red_weights_array_norm,
                                                     spec)

                            if line == 'oii':

                                # fit the gaussian to recover the parameters
                                gauss_values, covar = oii_gauss_fit(fit_wave_array,
                                                                    new_flux,
                                                                    redshift,
                                                                    f_red_weights_array_norm,
                                                                    central_wl,
                                                                    weight_fit)

                            else:

                                # fit the gaussian to recover the parameters
                                gauss_values, covar = gauss_fit(fit_wave_array,
                                                                new_flux,
                                                                f_red_weights_array_norm,
                                                                central_wl,
                                                                weight_fit)

                            # plt.show()

                            # append the returned values to the mc arrays
                            # only if the errors are less than tol

                            try:

                                amp_err = 100 * np.sqrt(covar[2][2]) / gauss_values['amplitude']

                                sig_err = 100 * np.sqrt(covar[0][0]) / gauss_values['sigma']

                                cen_err = 100 * np.sqrt(covar[1][1]) / gauss_values['center']

                            # if the error is thrown assign infinite errors
                            except TypeError:

                                amp_err = np.inf

                                sig_err = np.inf

                                cen_err = np.inf

                            if (amp_err < tol and sig_err < tol and cen_err < tol):

                                mc_sig_array.append(gauss_values['sigma'])
                                mc_amp_array.append(gauss_values['amplitude'])
                                mc_centre_array.append(gauss_values['center'])

                        # print 'This is how many survived %s' % len(mc_sig_array)
                        # np array the resultant mc arrays

                        mc_sig_array = np.array(mc_sig_array)
                        mc_amp_array = np.array(mc_amp_array)
                        mc_centre_array = np.array(mc_centre_array)

                        # make a histogram of the centre points and plot

    #                    hist, edges = np.histogram(mc_centre_array)
    #                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    #                    ax.plot(edges[:-1], hist)
    #                    plt.show()
    #                    plt.close('all')

                        c = 2.99792458E5

                        # make histograms of the MCMC results to
                        # determine the gaussian fitting parameters
                        # from a gaussian fit to those

                        vel_hist, vel_edges = np.histogram((c * ((mc_centre_array
                                                 - central_wl) / central_wl)),
                                                           bins=ntimes / 20.0)

                        sig_hist, sig_edges = np.histogram(c * (mc_sig_array / central_wl),
                                                           bins=ntimes / 20.0)

                        amp_hist, amp_edges = np.histogram(mc_amp_array, bins=ntimes / 20.0)

                        # make gaussian fits to these histograms

                        vel_gauss_values, vel_covar = mc_gauss_fit(vel_edges[:-1],
                                                                     vel_hist)

                        sig_gauss_values, sig_covar = mc_gauss_fit(sig_edges[:-1],
                                                                     sig_hist)

                        amp_gauss_values, amp_covar = mc_gauss_fit(amp_edges[:-1],
                                                                     amp_hist)

                        sn_array[i, j] = final_sn
                        rejection_array[i, j] = 1.5
                        vel_array[i, j] = vel_gauss_values['center']

                        if sig_gauss_values['center'] > 0 and sig_gauss_values['center'] < 3000:

                            disp_array[i, j] = sig_gauss_values['center']

                        else:

                            disp_array[i, j] = np.nan

                        flux_array[i, j] = amp_gauss_values['center']*spaxel_median_flux
                        flux_error_array[i, j] = amp_gauss_values['sigma']*spaxel_median_flux
                        vel_error_array[i, j] = vel_gauss_values['sigma']
                        sig_error_array[i, j] = sig_gauss_values['sigma']

                    elif (final_sn > 0 and final_sn < threshold) or \
                         (final_sn > threshold and (int_ratio < g_c_min or int_ratio > g_c_max)) or \
                         (final_sn > threshold and (amp_err > tol or sig_err > tol or cen_err > tol)):

                        # print 'Threshold reached but sum and gauss too disimilar'
                    
                        vel_array[i, j] = np.nan
                        disp_array[i, j] = np.nan
                        flux_array[i, j] = np.nan
                        flux_error_array[i, j] = np.nan
                        vel_error_array[i, j] = np.nan
                        sig_error_array[i, j] = np.nan

                        # populate the rejection array accordingly
                        if (final_sn > 0 and final_sn < threshold) and \
                           (int_ratio < g_c_min or int_ratio > g_c_max) and \
                           (amp_err > tol or sig_err > tol or cen_err > tol):
                           rejection_array[i, j] = 3
                        elif (final_sn > 0 and final_sn < threshold) and \
                             (int_ratio < g_c_min or int_ratio > g_c_max) and \
                             not (amp_err > tol or sig_err > tol or cen_err > tol):
                             rejection_array[i, j] = 4
                        elif (final_sn > 0 and final_sn < threshold) and \
                             not (int_ratio < g_c_min or int_ratio > g_c_max) and \
                             not (amp_err > tol or sig_err > tol or cen_err > tol):
                             rejection_array[i, j] = 5
                        elif not (final_sn > 0 and final_sn < threshold) and \
                                 (int_ratio < g_c_min or int_ratio > g_c_max) and \
                             not (amp_err > tol or sig_err > tol or cen_err > tol):
                             rejection_array[i, j] = 6
                        elif not (final_sn > 0 and final_sn < threshold) and \
                             not (int_ratio < g_c_min or int_ratio > g_c_max) and \
                                 (amp_err > tol or sig_err > tol or cen_err > tol):
                             rejection_array[i, j] = 7
                        elif not (final_sn > 0 and final_sn < threshold) and \
                                 (int_ratio < g_c_min or int_ratio > g_c_max) and \
                                 (amp_err > tol or sig_err > tol or cen_err > tol):
                             rejection_array[i, j] = 8
                        elif     (final_sn > 0 and final_sn < threshold) and \
                             not (int_ratio < g_c_min or int_ratio > g_c_max) and \
                                 (amp_err > tol or sig_err > tol or cen_err > tol):
                             rejection_array[i, j] = 9
                        else:
                             rejection_array[i, j] = 10

                    else:

                        # didn't reach target - store as nan

                        # print 'no improvement, stop trying to fix'

                        vel_array[i, j] = np.nan
                        disp_array[i, j] = np.nan
                        flux_array[i, j] = np.nan
                        flux_error_array[i, j] = np.nan
                        vel_error_array[i, j] = np.nan
                        sig_error_array[i, j] = np.nan
                        rejection_array[i, j] = 11

    # print 'This is the sigma error array: %s' % sig_error_array

    stdout.write('\n')

    # loop around noise array to clean up nan entries
    for i in range(0, len(noise_array)):
        for j in range(0, len(noise_array[0])):
            if np.isnan(noise_array[i][j]):
                # print 'Fixing nan value'
                noise_array[i][j] = np.nanmedian(noise_array)

    # print sn_array
    # plot all of the arrays

    try:

        vel_min, vel_max = np.nanpercentile(vel_array[mask_x_lower:mask_x_upper,
                                                      mask_y_lower:mask_y_upper],
                                            [15.0, 85.0])
    except TypeError:

        vel_min, vel_max = [-100, 100]

    try:

        sig_min, sig_max = np.nanpercentile(disp_array[mask_x_lower:mask_x_upper,
                                                       mask_y_lower:mask_y_upper],
                                            [15.0, 85.0])

    except TypeError:

        sig_min, sig_max = [0, 150]

    try:

        flux_min, flux_max = np.nanpercentile(flux_array[mask_x_lower:mask_x_upper,
                                                         mask_y_lower:mask_y_upper],
                                              [15.0, 85.0])

    except TypeError:

        flux_min, flux_max = [0, 5E-3]

    try:

        s_min, s_max = np.nanpercentile(signal_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [15.0, 85.0])

    except TypeError:


        s_min, s_max = [0, 0.01]

    try:

        sn_min, sn_max = np.nanpercentile(sn_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [15.0, 85.0])

    except TypeError:

        sn_min, sn_max = [0, 10]

    try:

        g_min, g_max = np.nanpercentile(measurement_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [15.0, 85.0])

    except TypeError:

        g_min, g_max = [0, 1.5]

    try:

        er_min, er_max = np.nanpercentile(vel_error_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [15.0, 85.0])

    except TypeError:

        er_min, er_max = [0, 100]

    try:

        sig_er_min, sig_er_max = np.nanpercentile(sig_error_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [15.0, 85.0])

    except TypeError:

        sig_er_min, sig_er_max = [0, 100]

    plt.close('all')

    # create 1x3 postage stamps of the different properties

    fig, ax = plt.subplots(1, 8, figsize=(30, 6))

    flux_cut = flux_array[mask_x_lower:mask_x_upper,
                          mask_y_lower:mask_y_upper]

    masked_flux_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_flux_array[i][j] = flux_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_flux_array[i][j] = masked_flux_array[i][j]

    im = ax[0].imshow(flux_array,
                      cmap=plt.get_cmap('jet'),
                      vmin=flux_min,
                      vmax=flux_max,
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[0])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # set the title
    ax[0].set_title('%s Flux' % line)

    vel_cut = vel_array[mask_x_lower:mask_x_upper,
                        mask_y_lower:mask_y_upper]

    masked_vel_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_vel_array[i][j] = vel_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_vel_array[i][j] = masked_vel_array[i][j]

    # now that we have the velocity array, can calculate the effects of
    # beam smearing on the sigma profile

    # ideally want to use the model velocity field to compute the
    # beam smearing. This is a bit cyclical as you need this
    # method to finish to compute the model - but most of the time
    # running this as a repeat. Therefore check for the existence of
    # the velocity field and a parameters list and use these to compute
    # the model field - if they don't exist.

    vel_field_name = incube[:-5] + line + '_vel_field.fits'

    params_name = incube[:-5] + line + '_vel_field_params_fixed.txt'

    # want to save the observed sigma, the resolution sigma,
    # the beam smeared sigma and the corrected intrinsic sigma
    # separately

    intrinsic_sigma = np.sqrt((disp_array)**2 -
                              sky_res_grid ** 2)


    # find the new total error - note once an error on the beam smearing
    # is understood more comprehensively would be good to include this
    # in the errors as well. Also I'm not sure this is the correct error
    # combination formula for combining things in quadrature

    total_sigma_error = np.sqrt(sig_error_array**2 + sky_res_error_grid**2)

    tot_sig_error_cut = total_sigma_error[mask_x_lower:mask_x_upper,
                                          mask_y_lower:mask_y_upper]

    masked_tot_sig_error_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_tot_sig_error_array[i][j] = tot_sig_error_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_tot_sig_error_array[i][j] = masked_tot_sig_error_array[i][j]

    # and find the cut version of all of these arrays

    int_sig_cut = intrinsic_sigma[mask_x_lower:mask_x_upper,
                                  mask_y_lower:mask_y_upper]

    masked_int_sig_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_int_sig_array[i][j] = int_sig_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_int_sig_array[i][j] = masked_int_sig_array[i][j]

    try:
        int_sig_min, int_sig_max = np.nanpercentile(masked_int_sig_array,
                                                    [15.0, 85.0])
    except TypeError:

        int_sig_min, int_sig_max = [50, 100]

    sky_res_cut = sky_res_grid[mask_x_lower:mask_x_upper,
                               mask_y_lower:mask_y_upper]

    masked_sky_res_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_sky_res_array[i][j] = sky_res_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_sky_res_array[i][j] = masked_sky_res_array[i][j]

    im = ax[1].imshow(masked_vel_array,
                      vmin=vel_min,
                      vmax=vel_max,
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[1])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # set the title
    ax[1].set_title('%s Velocity' % line)

    disp_cut = disp_array[mask_x_lower:mask_x_upper,
                          mask_y_lower:mask_y_upper]

    masked_disp_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_disp_array[i][j] = disp_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_disp_array[i][j] = masked_disp_array[i][j]

    disp_cut = vel_error_array[mask_x_lower:mask_x_upper,
                               mask_y_lower:mask_y_upper]

    masked_vel_error_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_vel_error_array[i][j] = disp_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_vel_error_array[i][j] = masked_vel_error_array[i][j]

    disp_cut = sn_array[mask_x_lower:mask_x_upper,
                               mask_y_lower:mask_y_upper]

    masked_sn_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_sn_array[i][j] = disp_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_sn_array[i][j] = masked_sn_array[i][j]

    disp_cut = noise_array[mask_x_lower:mask_x_upper,
                           mask_y_lower:mask_y_upper]

    masked_noise_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_noise_array[i][j] = disp_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_noise_array[i][j] = masked_noise_array[i][j]

    disp_cut = signal_array[mask_x_lower:mask_x_upper,
                           mask_y_lower:mask_y_upper]

    masked_signal_array = np.nan * np.empty(shape=(xpixs, ypixs))

    for i in range(xpixs):

        for j in range(ypixs):

            if (i >= mask_x_lower and i < mask_x_upper) \
               and (j >= mask_y_lower and j < mask_y_upper):

                masked_signal_array[i][j] = disp_cut[i - mask_x_lower][j - mask_y_lower]

            else:

                masked_signal_array[i][j] = masked_signal_array[i][j]

    im = ax[2].imshow(masked_int_sig_array,
                      vmin=int_sig_min,
                      vmax=int_sig_max,
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[2])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # set the title
    ax[2].set_title('%s Dispersion' % line)

    im = ax[3].imshow(signal_array,
                      vmin=s_min,
                      vmax=s_max,
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[3])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)


    # set the title
    ax[3].set_title('Signal Array')

    im = ax[4].imshow(sn_array,
                      vmin=sn_min,
                      vmax=sn_max,
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[4])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # set the title
    ax[4].set_title('sn array')

    im = ax[5].imshow(measurement_array,
                      vmin=g_min,
                      vmax=g_max,
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[5])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # set the title
    ax[5].set_title('sum over gauss')

    im = ax[6].imshow(vel_error_array,
                      vmin=er_min,
                      vmax=er_max,
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[6])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # set the title
    ax[6].set_title('Velocity Error')

    im = ax[7].imshow(masked_tot_sig_error_array,
                      vmin=sig_er_min,
                      vmax=sig_er_max,
                      cmap=plt.get_cmap('jet'),
                      interpolation='nearest')

    # add colourbar to each plot
    divider = make_axes_locatable(ax[7])
    cax_new = divider.append_axes('right', size='10%', pad=0.05)
    plt.colorbar(im, cax=cax_new)

    # set the title
    ax[7].set_title('Sigma Error')

    # plt.show()

    fig.savefig('%s_%s_stamps_gauss%s_t%s_%s_%s.pdf' % (incube[:-5],
                                                        line,
                                                        str(tol),
                                                        str(threshold),
                                                        method,
                                                        noise_method))

    plt.close('all')

    # also want to return the velocity error array and the velocity
    # array as fits files so they can be loaded into disk
    flux_hdu = fits.PrimaryHDU(masked_flux_array)

    flux_hdu.writeto('%s_%s_flux_field.fits' % (incube[:-5],
                                                line),
                     clobber=True)

    flux_error_hdu = fits.PrimaryHDU(flux_error_array)

    flux_error_hdu.writeto('%s_%s_flux_error_field.fits' % (incube[:-5],
                                                            line),
                           clobber=True)

    vel_hdu = fits.PrimaryHDU(masked_vel_array)

    vel_hdu.writeto('%s_%s_vel_field.fits' % (incube[:-5],
                                              line),
                     clobber=True)

    vel_err_hdu = fits.PrimaryHDU(vel_error_array)

    vel_err_hdu.writeto('%s_%s_error_field.fits'  % (incube[:-5],
                                                     line),
                        clobber=True)

    sig_hdu = fits.PrimaryHDU(masked_disp_array)

    sig_hdu.writeto('%s_%s_sig_field.fits' %   (incube[:-5],
                                                 line),
                     clobber=True)

    sig_int_hdu = fits.PrimaryHDU(masked_int_sig_array)

    sig_int_hdu.writeto('%s_%s_int_sig_field.fits'  % (incube[:-5],
                                                       line),
                        clobber=True)

    sig_sky_hdu = fits.PrimaryHDU(masked_sky_res_array)

    sig_sky_hdu.writeto('%s_%s_sig_sky_field.fits'  % (incube[:-5],
                                                       line),
                        clobber=True)

    sig_error_hdu = fits.PrimaryHDU(masked_tot_sig_error_array)

    sig_error_hdu.writeto('%s_%s_sig_error_field.fits'  % (incube[:-5],
                                                        line),
                          clobber=True)

    signal_hdu = fits.PrimaryHDU(signal_array)

    signal_hdu.writeto('%s_%s_signal_field.fits'  % (incube[:-5],
                                                     line),
                          clobber=True)

    noise_hdu = fits.PrimaryHDU(noise_array)

    noise_hdu.writeto('%s_%s_noise_field.fits'  % (incube[:-5],
                                                     line),
                          clobber=True)

    rejection_hdu = fits.PrimaryHDU(rejection_array)

    rejection_hdu.writeto('%s_%s_rejection_field.fits'  % (incube[:-5],
                                                     line),
                          clobber=True)

    sum_gauss_hdu = fits.PrimaryHDU(measurement_array)

    sum_gauss_hdu.writeto('%s_%s_sum_gauss_field.fits'  % (incube[:-5],
                                                           line),
                          clobber=True)

    sn_hdu = fits.PrimaryHDU(sn_array)

    sn_hdu.writeto('%s_%s_sn_field.fits'  % (incube[:-5],
                                             line),
                    clobber=True)

    # return the noise, signal and flux arrays for potential
    # voronoi binning
    
    return [flux_array,
            masked_vel_array,
            masked_int_sig_array,
            signal_array,
            noise_array,
            sn_array,
            measurement_array,
            masked_vel_error_array,
            masked_tot_sig_error_array]

def bin_three_for_noise(data):

    """
    Def:
    method to compute the noise after binning 3x3. Create a 3x3 binned
    entire cube and then compute the noise in the exact same way as
    in the standard case and compare. See whether this is better than the
    current compute noise reduction factor method!
    """
    three_binned_data = copy(data)
    for i in range(7, data.shape[1]-7):
        for j in range(7, data.shape[2]-7):
            stack_array = []
            # median stack the spectra 
            for a in range(i - 1, i + 2):
                for b in range(j - 1, j + 2):
                    stack_array.append(data[:, a, b])

            spec = np.nanmedian(stack_array, axis=0)
            three_binned_data[:,i,j] = spec

    return three_binned_data

def bin_five_for_noise(data):

    """
    Def:
    method to compute the noise after binning 3x3. Create a 3x3 binned
    entire cube and then compute the noise in the exact same way as
    in the standard case and compare. See whether this is better than the
    current compute noise reduction factor method!
    """
    five_binned_data = copy(data)
    for i in range(7, data.shape[1]-7):
        for j in range(7, data.shape[2]-7):
            stack_array = []
            # median stack the spectra 
            for a in range(i - 2, i + 3):
                for b in range(j - 2, j + 3):
                    stack_array.append(data[:, a, b])

            spec = np.nanmedian(stack_array, axis=0)
            five_binned_data[:,i,j] = spec

    return five_binned_data

def noise_from_mask_poly_subtract(cube_filter,
                                  data,
                                  mask_x_lower,
                                  mask_x_upper,
                                  mask_y_lower,
                                  mask_y_upper):

    """
    Def:
    *Helper function for the vel_field_mask_noise method*
    Compute the noise level in a datacube by examining the pixels which
    are not contaminated by the object - using exactly the same wavelength
    pixels which were used to look at the [OIII] flux. Then fit a polynomial
    to the stacked spectrum and subtract from the data - the point here is
    to get rid of the pedastal atop which the gaussian sits around the
    [OIII] line

    Input:
            data - full datacube from stacked object
            mask_x_lower - lower spatial dimension in x-direction
            mask_x_upper - upper spatial dimension in x-direction
            mask_y_lower - lower spatial dimension in y-direction
            mask_y_upper - upper spatial dimension in y direction

    Output:
            noise - single value, which is the noise for the spaxel
                    in consideration in the vel_field_mask_noise method

    """
    # create list to house the data from the unmasked pixels

    noise_list = []
    noise_values = []

    print 'printing mask values'
    print mask_x_lower, mask_x_upper, mask_y_lower, mask_y_upper
    print data.shape[2]

    # loop round and append to this list
    # four different mask segments to append

    for i in range(4, mask_x_lower + 1):

        for j in range(4, data.shape[2] - 4):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_upper, data.shape[1] - 4):

        for j in range(4, data.shape[2] - 4):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_lower, mask_x_upper + 1):

        for j in range(4, mask_y_lower + 1):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_lower, mask_x_upper + 1):

        for j in range(mask_y_upper, data.shape[2] - 4):

            noise_list.append(data[:, i, j])

    print 'noise list'
    print data.shape

    # polynomial fit to the noise spectrum (for continuum subtraction)

    poly_noise = np.nanmedian(noise_list, axis=0)
    x = np.arange(0, len(poly_noise), 1)

    if cube_filter == 'K':

        poly_mod = PolynomialModel(5)
        pars = poly_mod.guess(poly_noise[100:1600], x=x[100:1600])
        out = poly_mod.fit(poly_noise[100:1600], pars, x=x[100:1600])
        poly_best = out.eval(x=x)

    elif cube_filter == 'HK':

        poly_mod = PolynomialModel(5)
        pars = poly_mod.guess(poly_noise[100:1900], x=x[100:1900])
        out = poly_mod.fit(poly_noise[100:1900], pars, x=x[100:1900])
        poly_best = out.eval(x=x)

    elif cube_filter == 'H':

        poly_mod = PolynomialModel(5)
        pars = poly_mod.guess(poly_noise[100:1900], x=x[100:1900])
        out = poly_mod.fit(poly_noise[100:1900], pars, x=x[100:1900])
        poly_best = out.eval(x=x)

    else:

        poly_mod = PolynomialModel(5)
        pars = poly_mod.guess(poly_noise[100:1800], x=x[100:1800])
        out = poly_mod.fit(poly_noise[100:1800], pars, x=x[100:1800])
        poly_best = out.eval(x=x)


    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    ax.plot(x[100:1800], poly_noise[100:1800])
    ax.plot(x[100:1800], poly_best[100:1800])
    # plt.show()
    plt.close('all')

    # return the vector containing the 'thermal noise'
    # all of the signal to noise should be done relative
    # to this

    return poly_best

def empirically_mask_skylines(data,
                              mask_x_lower,
                              mask_x_upper,
                              mask_y_lower,
                              mask_y_upper):

    """
    Def:
    *Helper function for the vel_field_mask_noise method*
    Compute the noise level in a datacube by examining the pixels which
    are not contaminated by the object - using exactly the same wavelength
    pixels which were used to look at the [OIII] flux.

    Input:
            data - full datacube from stacked object
            lower_l - lower wavelength limit
            upper_l - upper wavelength limit
            mask_x_lower - lower spatial dimension in x-direction
            mask_x_upper - upper spatial dimension in x-direction
            mask_y_lower - lower spatial dimension in y-direction
            mask_y_upper - upper spatial dimension in y direction

    Output:
            noise - single value, which is the noise for the spaxel
                    in consideration in the vel_field_mask_noise method

    """
    # create list to house the data from the unmasked pixels

    noise_list = []


    # loop round and append to this list
    # four different mask segments to append

    for i in range(7, mask_x_lower + 1):

        for j in range(7, data.shape[2] - 7):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_upper, data.shape[1] - 7):

        for j in range(7, data.shape[2] - 7):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_lower, mask_x_upper + 1):

        for j in range(7, mask_y_lower + 1):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_lower, mask_x_upper + 1):

        for j in range(mask_y_upper, data.shape[2] - 7):

            noise_list.append(data[:, i, j])

    # now get a single spectrum by median stacking everything in this
    # noise list

    median_noise_spec = np.nanmedian(noise_list, axis=0)
#    fig, ax = plt.subplots(1, 1, figsize=(14,8))
#    wave = np.arange(0,len(median_noise_spec))
#    ax.plot(wave,median_noise_spec)
    stand_dev = astmad(median_noise_spec[100:1800],ignore_nan=True)
    indices = np.logical_or(median_noise_spec > 5*stand_dev, median_noise_spec < -5*stand_dev)
#    median_noise_spec[indices] = np.nan
#    ax.plot(wave,median_noise_spec,color='red',lw=2)
#    plt.show()
#    plt.close('all')

    return indices


def noise_from_mask(data,
                    lower_l,
                    upper_l,
                    mask_x_lower,
                    mask_x_upper,
                    mask_y_lower,
                    mask_y_upper):

    """
    Def:
    *Helper function for the vel_field_mask_noise method*
    Compute the noise level in a datacube by examining the pixels which
    are not contaminated by the object - using exactly the same wavelength
    pixels which were used to look at the [OIII] flux.

    Input:
            data - full datacube from stacked object
            lower_l - lower wavelength limit
            upper_l - upper wavelength limit
            mask_x_lower - lower spatial dimension in x-direction
            mask_x_upper - upper spatial dimension in x-direction
            mask_y_lower - lower spatial dimension in y-direction
            mask_y_upper - upper spatial dimension in y direction

    Output:
            noise - single value, which is the noise for the spaxel
                    in consideration in the vel_field_mask_noise method

    """
    # create list to house the data from the unmasked pixels

    noise_list = []
    noise_values = []
    p_noise_values = []

    # loop round and append to this list
    # four different mask segments to append

    for i in range(7, mask_x_lower + 1):

        for j in range(7, data.shape[2] - 7):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_upper, data.shape[1] - 7):

        for j in range(7, data.shape[2] - 7):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_lower, mask_x_upper + 1):

        for j in range(7, mask_y_lower + 1):

            noise_list.append(data[:, i, j])

    for i in range(mask_x_lower, mask_x_upper + 1):

        for j in range(mask_y_upper, data.shape[2] - 7):

            noise_list.append(data[:, i, j])

    # now for every entry in the noise list compute the noise
    # and append to the noise_values list

    for entry in noise_list:

        noise_values.append(np.nansum(entry[lower_l:upper_l]))
        p_noise_values.append(astmad(entry[lower_l:upper_l],ignore_nan=True))

    noise_values = np.array(noise_values)

    hist, edges = np.histogram(noise_values, bins=20)

    # fig, ax = plt.subplots(1, 1, figsize=(10,10))

    # ax.plot(edges[:-1], hist)
    # ax.set_title('distribution of noise estimates')

    # print 'distribution of noise estimates %s' % len(noise_values)

    # plt.show()
    # plt.close('all')

    # what to do with these independent noise estimates?
    # will take the median for now but could also take the dispersion
    #print 'NOISE VALUES: %s' % np.sort(noise_values)
    final_noise = astmad(noise_values,ignore_nan=True)
    final_p_noise = np.nanmedian(p_noise_values)

    return final_noise, final_p_noise

def binning_three(data,
                  noise_value,
                  i,
                  j,
                  lower_lim,
                  upper_lim,
                  red_factor,
                  method):

    """
    Def: Helper method to do the 3x3 spatial binning for the stott
    velocity field function.

    Input:
            data - datacube from the object
            i - spaxel under consideration in the stott loop
            j - same as above
            lower_lim - lower limit to truncate the spectrum
            upper_lim - upper, for the signal of the line computation
            method - method to use to stack the spectra together after

    Output:
            spec - stacked spectrum for spaxel i, j of length lower_lim
                    + upper_lim (i.e. truncated between these limits)
    """

    # first construct loop over the i - 1 - i +1 and same for jet
    # need except statement incase the cube_boundary is reached

    stack_array = []

    try:

        for a in range(i - 1, i + 2):

            for b in range(j - 1, j + 2):

                stack_array.append(data[:, a, b])

        if method == 'median':

            spec = np.nanmedian(stack_array, axis=0)

            new_noise_value = noise_value / red_factor

        elif method == 'sum':

            spec = np.nansum(stack_array, axis=0)

            new_noise_value = (9.0 / red_factor) * noise_value

        elif method == 'mean':

            spec = np.nanmean(stack_array, axis=0)

            new_noise_value = noise_value / red_factor

        else:

            raise ValueError('Please choose a valid stacking method')

    except IndexError:

        # print 'encountered the cube boundary'

        spec = data[:, i, j]

        new_noise_value = noise_value

    return spec, new_noise_value

def binning_five(data,
                 noise_value,
                 i,
                 j,
                 lower_lim,
                 upper_lim,
                 red_factor,
                 method):

    """
    Def: Helper method to do the 5x5 spatial binning for the stott
    velocity field function.

    Input:
            data - datacube from the object
            noise_value - the noise associated with that spaxel
            i - spaxel under consideration in the stott loop
            j - same as above
            lower_lim - lower limit to truncate the spectrum
            upper_lim - upper, for the signal of the line computation
            method - method to use to stack the spectra together after

    Output:
            spec - stacked spectrum for spaxel i, j of length lower_lim
                    + upper_lim (i.e. truncated between these limits)
    """

    # first construct loop over the i - 1 - i +1 and same for jet
    # need except statement incase the cube_boundary is reached

    stack_array = []

    try:

        for a in range(i - 2, i + 3):

            for b in range(j - 2, j + 3):

                stack_array.append(data[:, a, b])

        if method == 'median':

            spec = np.nanmedian(stack_array, axis=0)

            new_noise_value = noise_value / red_factor

        elif method == 'sum':

            spec = np.nansum(stack_array, axis=0)

            new_noise_value = (25.0 / red_factor) * noise_value

        elif method == 'mean':

            spec = np.nanmean(stack_array, axis=0)

            new_noise_value = noise_value / red_factor

        else:

            raise ValueError('Please choose a valid stacking method')

    except IndexError:

        # print 'encountered the cube boundary'

        spec = data[:, i, j]

        new_noise_value = noise_value

    # compute the new noise value

    return spec, new_noise_value

def mc_gauss_fit(fit_wl,
                 fit_flux):

    """
    Def:
    Performs simple gaussian fit, guessing initial parameters from the data
    and given input wavelength and input flux values

    Input:
            fit_wl - wavelength of spectrum to fit
            fit_flux - flux of spectrum to fitsWavelength

    Output:
            fit_params - dictionary containing the best fit parameters
                        for each of the spectra
    """

    # construct gaussian model using lmfit

    gmod = GaussianModel()

    # set the initial parameter values

    pars = gmod.guess(fit_flux, x=fit_wl)

    # perform the fit
    out = gmod.fit(fit_flux,
                   pars,
                   x=fit_wl)

    # print the fit report
#        print out.fit_report()
#        # plot to make sure things are working
#        fig, ax = plt.subplots(figsize=(14, 6))
#        ax.plot(fit_wl, fit_flux, color='blue')
#        ax.plot(fit_wl, out.best_fit, color='red')
#        plt.show()
#        plt.close('all')

    return out.best_values, out.covar


def compute_noise_reduction_factor_three(data,
                                         xpixs,
                                         ypixs,
                                         lower_limit,
                                         upper_limit):

    """
    Def: compute the factor by which the noise goes down when binning.
    I will use this value regardless of using the noise cube or the mask
    as all the results seem to be converging towards the same vel maps
    thankfully.

    Input:

            line_noise - the computed line noise for that spaxel_noise
            data - the full data array for the cube
            lower_limit - lower_limit for the spaxel reading
            upper-limit - upper wavelength limit for the spaxel reading

    Output: value indicating how much the noise should reduce by

    """

    # will do this by examining 4 different sections of the cubes
    # in the four corners and then one in the center
    # then taking the median of the reduction factors

    factor_list = []

    # first section in the top left
    # find the single noise array

    line_array = data[lower_limit + 20:upper_limit + 20, 13, 13]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(12, 15):

        for h in range(12, 15):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # Second section in the bottom left
    # find the single noise array

    line_array = data[lower_limit + 20:upper_limit + 20,
                      xpixs - 13,
                      13]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(xpixs - 14, xpixs - 11):

        for h in range(12, 15):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the oiii line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # Third section in the bottom right
    # find the single noise array

    line_array = data[lower_limit + 20:upper_limit + 20,
                      xpixs - 13,
                      ypixs - 13]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(xpixs - 14, xpixs - 11):

        for h in range(ypixs - 14, ypixs - 11):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the oiii line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # fourth section in top right

    line_array = data[lower_limit + 20:upper_limit + 20,
                      13,
                      ypixs - 13]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(12, 15):

        for h in range(ypixs - 14, ypixs - 11):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the oiii line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # print 'NOISE REDUCTION FACTOR: %s' % np.nanmedian(factor_list)

    return np.nanmedian(factor_list)

def compute_noise_reduction_factor_five(data,
                                        xpixs,
                                        ypixs,
                                        lower_limit,
                                        upper_limit):

    """
    Def: compute the factor by which the noise goes down when binning.
    I will use this value regardless of using the noise cube or the mask
    as all the results seem to be converging towards the same vel maps
    thankfully.

    Input:

            line_noise - the computed line noise for that spaxel_noise
            data - the full data array for the cube
            lower_limit - lower_limit for the spaxel reading
            upper-limit - upper wavelength limit for the spaxel reading

    Output: value indicating how much the noise should reduce by

    """

    # will do this by examining 4 different sections of the cubes
    # in the four corners and then one in the center
    # then taking the median of the reduction factors

    factor_list = []

    # first section in the top left
    # find the single noise array

    line_array = data[lower_limit + 20:upper_limit + 20, 14, 14]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(12, 17):

        for h in range(12, 17):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the oiii line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # Second section in the bottom left
    # find the single noise array

    line_array = data[lower_limit + 20:upper_limit + 20,
                      xpixs - 14,
                      14]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(xpixs - 17, xpixs - 12):

        for h in range(12, 17):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the oiii line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # Third section in the bottom right
    # find the single noise array

    line_array = data[lower_limit + 20:upper_limit + 20,
                      xpixs - 14,
                      ypixs - 14]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(xpixs - 17, xpixs - 12):

        for h in range(ypixs - 17, ypixs - 12):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the oiii line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # fourth section in top right

    line_array = data[lower_limit + 20:upper_limit + 20,
                      14,
                      ypixs - 14]

    # find the standard deviation of the noise array

    line_noise = np.nanstd(line_array, axis=0)

    # initiate the 3x3 noise array

    treb_list = []

    for g in range(12, 17):

        for h in range(ypixs - 17, ypixs - 12):

            treb_list.append(data[:, g, h])

    treb_noise_new = np.nanmedian(treb_list, axis=0)

    # get away from the oiii line by adding 20 to the indices

    treb_array = treb_noise_new[lower_limit + 20:upper_limit + 20]

    factor_list.append(line_noise / np.nanstd(treb_array, axis=0))

    # print 'NOISE REDUCTION FACTOR: %s' % np.nanmedian(factor_list)

    return np.nanmedian(factor_list)

def ped_gauss_fit(fit_wl, fit_flux):

    """
    Def:
    Performs simple gaussian fit, guessing initial parameters from the data
    and given input wavelength and input flux values - this time the base
    of the gaussian is left as a free parameter

    Input:
            fit_wl - wavelength of spectrum to fit
            fit_flux - flux of spectrum to fitsWavelength

    Output:
            fit_params - dictionary containing the best fit parameters
                        for each of the spectra
    """

    # construct gaussian model using lmfit

    gmod = GaussianModel()

    # construct constant model using lmfit

    cmod = ConstantModel()

    # set the initial parameter values

    pars = gmod.guess(fit_flux, x=fit_wl)

    # add an initial guess at the constant (0)

    pars += cmod.make_params(c=0)

    mod = cmod + gmod

    # perform the fit
    out = mod.fit(fit_flux, pars, x=fit_wl)

    # print the fit report
    # print out.fit_report()

    # plot to make sure things are working
#        fig, ax = plt.subplots(figsize=(14, 6))
#        ax.plot(fit_wl, fit_flux, color='blue')
#        ax.plot(fit_wl, out.best_fit, color='red')
#        plt.show()
#        plt.close('all')

    return out.best_values, out.covar


def gauss_fit(fit_wl,
              fit_flux,
              weights,
              central_wl,
              weight_fit):

    """
    Def:
    Performs simple gaussian fit, guessing initial parameters from the data
    and given input wavelength and input flux values

    Input:
            fit_wl - wavelength of spectrum to fit
            fit_flux - flux of spectrum to fitsWavelength

    Output:
            fit_params - dictionary containing the best fit parameters
                        for each of the spectra
    """
    # at the moment fitting the lines without skyline weighting
    # construct gaussian model using lmfit

    gmod = GaussianModel(missing='drop')

    # set the initial parameter values

    pars = gmod.guess(fit_flux, x=fit_wl)

    if weight_fit:
        # perform the fit
        out = gmod.fit(fit_flux,
                       pars,
                       x=fit_wl,
                       weights=(1/weights)**2)
#        print out.fit_report()
#        res_plot = out.plot()
#        plt.show(res_plot)
#        plt.close(res_plot)
    else:
        # perform the fit
        out = gmod.fit(fit_flux,
                       pars,
                       x=fit_wl)
#        print out.fit_report()
#        res_plot = out.plot()
#        plt.show(res_plot)
#        plt.close(res_plot)

#    fig = out.plot()
#    plt.show(fig)
#    plt.close('all')
#    print out.fit_report()

    # print the fit report
    # print out.fit_report()

    # plot to make sure things are working
#    fig, ax = plt.subplots(figsize=(14, 6))
#    ax.plot(fit_wl, fit_flux, color='blue')
#    ax.plot(fit_wl, out.best_fit, color='red')
#    plt.show()
#    plt.close('all')

    return out.best_values, out.covar

def oii_gauss_fit(fit_wl,
                  fit_flux,
                  redshift,
                  weights,
                  central_wl,
                  weight_fit):

    # define the line wavelengths
    oii_3727_rest = 0.372709
    oii_3727_shifted = (1 + redshift) * oii_3727_rest

    # note we'll use an expression to define the position of this
    # rather than defining minimum and max and shifted
    oii_3729_rest = 0.372988
    oii_3729_shifted = (1 + redshift) * oii_3729_rest

    # separation between the two
    delta_oii = oii_3729_rest - oii_3727_rest

    gmod = GaussianModel()

    pars = gmod.guess(fit_flux, x=fit_wl)

    sig_guess = pars.valuesdict()['sigma']/2.
    amp_guess = pars.valuesdict()['amplitude']/2.

    # construct a composite gaussian model with prefix parameter names
    comp_mod = GaussianModel(missing='drop',
                             prefix='oiil_') + \
               GaussianModel(missing='drop',
                             prefix='oiih_')
    # set the wavelength value with min and max range
    # and initialise the other parameters

    comp_mod.set_param_hint('oiil_center',
                            value=oii_3727_shifted)
    comp_mod.set_param_hint('oiih_center',
                            value=oii_3729_shifted,
                            expr='((%.6f)*%.6f) + oiil_center' % (1.+redshift,delta_oii))
    comp_mod.set_param_hint('oiil_amplitude',
                            value=amp_guess,
                            min=0)
    comp_mod.set_param_hint('oiih_amplitude',
                            value=amp_guess,
                            expr='1.*oiil_amplitude')
    comp_mod.set_param_hint('oiil_sigma',
                            value=sig_guess)
    comp_mod.set_param_hint('oiih_sigma',
                            value=sig_guess,
                            expr='1.*oiil_sigma')

    pars = comp_mod.make_params()

    if weight_fit:
        # perform the fit
        out = comp_mod.fit(fit_flux,
                           pars,
                           x=fit_wl,
                           weights=(1/weights)**2)
#        print out.fit_report()
#        res_plot = out.plot()
#        plt.show(res_plot)
#        plt.close(res_plot)
    else:
        # perform the fit
        out = comp_mod.fit(fit_flux,
                           pars,
                           x=fit_wl)
#        print out.fit_report()
#        res_plot = out.plot()
#        plt.show(res_plot)
#        plt.close(res_plot)

    # now try to salvage the best values dictionary

    gauss_values = {'amplitude': out.best_values['oiil_amplitude'] + out.best_values['oiih_amplitude'],
                    'sigma': out.best_values['oiil_sigma'],
                    'center': (out.best_values['oiil_center'] + out.best_values['oiih_center'])/2.0 }

    new_covar = copy(out.covar)
    try:
        new_covar[0][0] = out.covar[2][2]
        new_covar[1][1] = out.covar[0][0]
        new_covar[2][2] = out.covar[1][1]
        return gauss_values, new_covar
    except TypeError:
        return gauss_values, new_covar

def noise_from_masked_spectrum(wave_array,
                               spec,
                               redshift,
                               region_length,
                               spec_index,
                               filt,
                               prog):
    
    """
    Def:
    Calculate the noise in every spaxel by masking the sky lines and
    the emission lines and fitting a gaussian to the histogram of the
    remaining flux values
    """

    # first mask the sky

    wave_array, sky_masked_spec = mask_the_sky.masking_sky(wave_array,
                                                           spec,
                                                           filt)

    # now mask the emission lines

    sky_and_emission_line_masked_spec = mask_emission_lines(wave_array,
                                                            sky_masked_spec,
                                                            redshift,
                                                            filt,
                                                            prog)

    # plot

#    fig, ax = plt.subplots(1, 1, figsize=(16,10))
#    ax.plot(wave_array,sky_and_emission_line_masked_spec)
#    plt.show()
#    plt.close('all')

    # now going to step out from the central wavelength point and
    # look for unmasked spectral pixels, appending these to a signal
    # array, computing the signal from them and appending to a 
    # standard deviation array from which the noise will be calculated

    std_array = []
    search_index = 1
    while len(std_array) < 8:
        signal_array = []
        while len(signal_array) < region_length:
            low_index = spec_index - search_index
            # only consider 'good' spectral range
            if low_index > 100:
                if sky_and_emission_line_masked_spec[low_index]:
                    signal_array.append(sky_and_emission_line_masked_spec[low_index])
            high_index = spec_index + search_index
            if high_index < 1750:
                if sky_and_emission_line_masked_spec[high_index]:
                    signal_array.append(sky_and_emission_line_masked_spec[high_index])
            search_index += 1
        std_array.append(np.nansum(signal_array))
    
    return np.std(std_array)

def noise_from_histogram(wave_array,
                         spec,
                         redshift,
                         filt,
                         prog):
    
    """
    Def:
    Calculate the noise in every spaxel by masking the sky lines and
    the emission lines and fitting a gaussian to the histogram of the
    remaining flux values
    """

    # first mask the sky

    wave_array, sky_masked_spec = mask_the_sky.masking_sky(wave_array,
                                                           spec,
                                                           filt)

    # now mask the emission lines

    sky_and_emission_line_masked_spec = mask_emission_lines(wave_array,
                                                            sky_masked_spec,
                                                            redshift,
                                                            filt,
                                                            prog)

    # plot

#    fig, ax = plt.subplots(1, 1, figsize=(16,10))
#    ax.plot(wave_array,sky_and_emission_line_masked_spec)
#    plt.show()
#    plt.close('all')

    # recover the compressed spectrum

    compressed_spec = sky_and_emission_line_masked_spec[100:1700].compressed()
    compressed_spec = compressed_spec[np.logical_or(compressed_spec >=0,compressed_spec<0)]

    # take only the 5th to 95th percentile
    compressed_spec = np.sort(compressed_spec)
    compressed_spec = compressed_spec[int(len(compressed_spec)/20.0):int(0.95*len(compressed_spec))]

    bins, centres = np.histogram(compressed_spec, bins=20)
    noise_mod = GaussianModel(missing='drop')
    noise_pars = noise_mod.guess(bins, x=centres[:-1])
    noise_result = noise_mod.fit(bins, params=noise_pars, x=centres[:-1])

    #print noise_result.fit_report()
    #print bins, centres
#    noise_best_fit = noise_result.eval(x=centres[:-1])
#    fig, ax = plt.subplots(1,1,figsize=(8,10))
#    ax.scatter(centres[:-1],bins)
#    ax.plot(centres[:-1],noise_best_fit)
#    plt.show()
#    plt.close('all')

    noise = noise_result.best_values['sigma']
    return noise, noise_result.best_values['center']


def sky_res(sky_flux,
            sky_wave,
            sky_x_dim,
            sky_y_dim,
            llow,
            lhigh):

    """
    Def:
    Fit a skyline to determine the instrumental resolution.
    """

    sky_indices = np.where(np.logical_and(sky_wave > llow,
                                          sky_wave < lhigh))[0]

    sky_gauss_wave = sky_wave[sky_indices]

    sky_gauss_flux = sky_flux[:,
                              np.round(sky_x_dim / 2.0),
                              np.round(sky_y_dim / 2.0)][sky_indices]

    # plug these into the gaussian fitting routine

    gauss_values, covar = ped_gauss_fit(sky_gauss_wave,
                                        sky_gauss_flux)

    return 2.99792458E5 * (gauss_values['sigma'] / gauss_values['center'])

def continuum_subtract_full(flux_array,
                            wave_array,
                            redshift,
                            filt,
                            prog):

    """
    Def:
    Fit a polynomial to the noise spectrum, for the purpose of subtracting
    the thermal noise at the long wavelength end of the K-band.
    The noise spectrum is the sum of all spaxels not in the object spectrum,
    which are more than 5 pixels from the cube border.

    Input: 
            wave_array - the wavelength array of the spectrum
            flux_array - the summed spectrum

    Output:
            continuum subtracted final spectrum

    """

    # first mask the sky

    wave_array, sky_masked_spec = mask_the_sky.masking_sky(wave_array,
                                                           flux_array,
                                                           filt)

    # now mask the emission lines

    sky_and_line_masked_spec = mask_emission_lines(wave_array,
                                                   sky_masked_spec,
                                                   redshift,
                                                   filt,
                                                   prog)
    # now do a running median - want to fit to this
    # rather than to the actual noisy data
    # depends on filt and fit_filt combination 
    bins = np.linspace(wave_array[100], wave_array[1800], 50)
    delta = bins[1] - bins[0]
    idx = np.digitize(wave_array, bins)
    running_median = [np.nanmedian(sky_and_line_masked_spec[idx==k]) for k in range(50)]

    xl = 100
    xu = 1800

    # use lmfit to define the model and do the fitting
    poly_mod = PolynomialModel(7,missing='drop')
    pars = poly_mod.make_params()

    # for the masked array to work need to assign the parameters

    pars['c0'].set(value=0)
    pars['c1'].set(value=0)
    pars['c2'].set(value=0)
    pars['c3'].set(value=0)
    pars['c4'].set(value=0)
    pars['c5'].set(value=0)
    pars['c6'].set(value=0)
    pars['c7'].set(value=0)

#    fig, ax = plt.subplots(1,1,figsize=(18,8))
#    ax.plot(wave_array[xl:xu],sky_and_line_masked_spec[xl:xu],drawstyle='steps-mid')
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
#    ax.plot(wave_array[100:1750],sky_and_line_masked_spec[100:1750],drawstyle='steps-mid')
#    ax.plot(wave_array[100:1750],poly_best[100:1750],drawstyle='steps-mid',color='red',lw=2)
#    plt.show()
#    plt.close('all')
    return poly_best

def mask_emission_lines(wave_array,
                        spec,
                        redshift,
                        filt,
                        prog):

    if prog == 'klp':
        
        # choose the appropriate sky dictionary for the filter
        if filt == 'YJ':

            oii_rest = 0.37284835
            oii_shifted = (1 + redshift) * oii_rest
            oii_shifted_index = np.nanargmin(np.abs(wave_array - oii_shifted))
            oii_shifted_min = wave_array[oii_shifted_index-20]
            oii_shifted_max = wave_array[oii_shifted_index+20]
            #print 'YJ-band emission lines to mask'

            # also return a total masked spectrum which blocks out
            # emission lines also

            wave_array_total = copy(wave_array)   
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > oii_shifted_min,
                               wave_array_total < oii_shifted_max),
                               wave_array_total, copy=True)
            spec_total = ma.MaskedArray(spec,
                                        mask=wave_array_total.mask)


        elif filt == 'H':

            h_beta_rest = 0.4862721
            h_beta_shifted = (1 + redshift) * h_beta_rest
            h_beta_shifted_index = np.nanargmin(np.abs(wave_array - h_beta_shifted))
            h_beta_shifted_min = wave_array[h_beta_shifted_index-20]
            h_beta_shifted_max = wave_array[h_beta_shifted_index+20]
            oiii_4960_rest = 0.4960295
            oiii_4960_shifted = (1 + redshift) * oiii_4960_rest
            oiii_4960_shifted_index = np.nanargmin(np.abs(wave_array - oiii_4960_shifted))
            oiii_4960_shifted_min = wave_array[oiii_4960_shifted_index-20]
            oiii_4960_shifted_max = wave_array[oiii_4960_shifted_index+20]
            oiii_5008_rest = 0.5008239
            oiii_5008_shifted = (1 + redshift) * oiii_5008_rest
            oiii_5008_shifted_index = np.nanargmin(np.abs(wave_array - oiii_5008_shifted))
            oiii_5008_shifted_min = wave_array[oiii_5008_shifted_index-20]
            oiii_5008_shifted_max = wave_array[oiii_5008_shifted_index+20]
            #print 'H-band emission lines to mask'

            wave_array_total = copy(wave_array)   
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > h_beta_shifted_min,
                               wave_array_total < h_beta_shifted_max),
                               wave_array_total, copy=True)
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > oiii_4960_shifted_min,
                               wave_array_total < oiii_4960_shifted_max),
                               wave_array_total, copy=True)
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > oiii_5008_shifted_min,
                               wave_array_total < oiii_5008_shifted_max),
                               wave_array_total, copy=True)
            spec_total = ma.MaskedArray(spec,
                                        mask=wave_array_total.mask)

        else:

            h_alpha_rest = 0.6564614
            h_alpha_shifted = (1 + redshift) * h_alpha_rest
            h_alpha_shifted_index = np.nanargmin(np.abs(wave_array - h_alpha_shifted))
            h_alpha_shifted_min = wave_array[h_alpha_shifted_index-20]
            h_alpha_shifted_max = wave_array[h_alpha_shifted_index+20]
            nii_rest = 0.658523
            nii_shifted = (1 + redshift) * nii_rest
            nii_shifted_index = np.nanargmin(np.abs(wave_array - nii_shifted))
            nii_shifted_min = wave_array[nii_shifted_index-20]
            nii_shifted_max = wave_array[nii_shifted_index+20]
            sii_lower_rest = 0.671829
            sii_lower_shifted = (1 + redshift) * sii_lower_rest
            sii_lower_shifted_index = np.nanargmin(np.abs(wave_array - sii_lower_shifted))
            sii_lower_shifted_min = wave_array[sii_lower_shifted_index-20]
            sii_lower_shifted_max = wave_array[sii_lower_shifted_index+20]
            sii_upper_rest = 0.671829
            sii_upper_shifted = (1 + redshift) * sii_upper_rest
            sii_upper_shifted_index = np.nanargmin(np.abs(wave_array - sii_upper_shifted))
            sii_upper_shifted_min = wave_array[sii_upper_shifted_index-20]
            sii_upper_shifted_max = wave_array[sii_upper_shifted_index+20]

            #print 'K-band emission lines to mask'

            wave_array_total = copy(wave_array)   
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > h_alpha_shifted_min,
                               wave_array_total < h_alpha_shifted_max),
                               wave_array_total, copy=True)
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > nii_shifted_min,
                               wave_array_total < nii_shifted_max),
                               wave_array_total, copy=True)
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > sii_lower_shifted_min,
                               wave_array_total < sii_lower_shifted_max),
                               wave_array_total, copy=True)
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > sii_upper_shifted_min,
                               wave_array_total < sii_upper_shifted_max),
                               wave_array_total, copy=True)
            spec_total = ma.MaskedArray(spec,
                                        mask=wave_array_total.mask)

    else:

        # choose the appropriate sky dictionary for the filter
        if filt == 'H':

            oii_rest = 0.37284835
            oii_shifted = (1 + redshift) * oii_rest
            oii_shifted_index = np.nanargmin(np.abs(wave_array - oii_shifted))
            oii_shifted_min = wave_array[oii_shifted_index-20]
            oii_shifted_max = wave_array[oii_shifted_index+20]
            #print 'YJ-band emission lines to mask'

            # also return a total masked spectrum which blocks out
            # emission lines also

            wave_array_total = copy(wave_array)   
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > oii_shifted_min,
                               wave_array_total < oii_shifted_max),
                               wave_array_total, copy=True)
            spec_total = ma.MaskedArray(spec,
                                        mask=wave_array_total.mask)

        else:

            h_beta_rest = 0.4862721
            h_beta_shifted = (1 + redshift) * h_beta_rest
            h_beta_shifted_index = np.nanargmin(np.abs(wave_array - h_beta_shifted))
            h_beta_shifted_min = wave_array[h_beta_shifted_index-20]
            h_beta_shifted_max = wave_array[h_beta_shifted_index+20]
            oiii_4960_rest = 0.4960295
            oiii_4960_shifted = (1 + redshift) * oiii_4960_rest
            oiii_4960_shifted_index = np.nanargmin(np.abs(wave_array - oiii_4960_shifted))
            oiii_4960_shifted_min = wave_array[oiii_4960_shifted_index-20]
            oiii_4960_shifted_max = wave_array[oiii_4960_shifted_index+20]
            oiii_5008_rest = 0.5008239
            oiii_5008_shifted = (1 + redshift) * oiii_5008_rest
            oiii_5008_shifted_index = np.nanargmin(np.abs(wave_array - oiii_5008_shifted))
            oiii_5008_shifted_min = wave_array[oiii_5008_shifted_index-20]
            oiii_5008_shifted_max = wave_array[oiii_5008_shifted_index+20]
            #print 'H-band emission lines to mask'

            wave_array_total = copy(wave_array)   
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > h_beta_shifted_min,
                               wave_array_total < h_beta_shifted_max),
                               wave_array_total, copy=True)
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > oiii_4960_shifted_min,
                               wave_array_total < oiii_4960_shifted_max),
                               wave_array_total, copy=True)
            wave_array_total = ma.masked_where(
                np.logical_and(wave_array_total > oiii_5008_shifted_min,
                               wave_array_total < oiii_5008_shifted_max),
                               wave_array_total, copy=True)
            spec_total = ma.MaskedArray(spec,
                                        mask=wave_array_total.mask)


    return spec_total

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

    # construct the new flux array
    ran_array = np.random.normal(scale=abs(noise), size=len(flux_array))

    # do the perturbation using a gaussian distributed value
    # with mean of the flux array and sigma of the noise value

    return ran_array + flux_array

#multi_vel_field_stott('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Hband/KLP_H_NAMES_FINAL.txt',
#                        'oiii',
#                          3.0,
#                          0.25,
#                          1.75,
#                          ntimes=200,
#                          spatial_smooth=True,
#                          spectral_smooth=False,
#                          smoothing_psf=0.2,
#                          spectral_smooth_width=2,
#                          prog='klp',
#                          emp_mask=True,
#                          weight_fit=False)