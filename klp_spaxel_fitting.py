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

        noise_method = 'cube'

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
                                noise_method=noise_method)

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
                            tol=30,
                            method='median',
                            noise_method='cube',
                            ntimes=200):

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

    # define the skyline weights array
    if cube.filter == 'K':
        weights_file = incube[:-5] + '_k_sky.fits'
    elif cube.filter == 'H':
        weights_file = incube[:-5] + '_h_sky.fits'
    elif cube.filter == 'YJ':
        weights_file = incube[:-5] + '_yj_sky.fits'
    else:
        weights_file = incube[:-5] + '_NONONO_sky.fits'

    weights_array = fits.open(weights_file)[0].data

    skycube = cubeOps(sky_cube)

    sky_wave = skycube.wave_array

    sky_data = skycube.data

    sky_x_dim = sky_data.shape[1]

    sky_y_dim = sky_data.shape[2]

    data = cube.data

    noise = cube.Table[2].data

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

    # get the wavelength array

    wave_array = cubeOps(incube).wave_array

    if line == 'oiii':

        central_wl = 0.500824 * (1. + redshift)

    elif line == 'hb':

        central_wl = 0.486268 * (1. + redshift)

    elif line == 'oii':

        central_wl = 0.37275 * (1. + redshift)

    elif line == 'ha':

        central_wl = 0.6564614 * (1. + redshift)

    # find the index of the chosen emission line
    line_idx = np.argmin(np.abs(wave_array - central_wl))

    # the shape of the data is (spectrum, xpixel, ypixel)
    # loop through each x and y pixel and get the OIII5007 S/N

    sn_array = np.empty(shape=(xpixs, ypixs))

    signal_array = np.empty(shape=(xpixs, ypixs))

    noise_array = np.empty(shape=(xpixs, ypixs))

    vel_array = np.empty(shape=(xpixs, ypixs))

    disp_array = np.empty(shape=(xpixs, ypixs))

    flux_array = np.empty(shape=(xpixs, ypixs))

    vel_error_array = np.empty(shape=(xpixs, ypixs))

    sig_error_array = np.empty(shape=(xpixs, ypixs))

    # array to check the coincidence of gauss fit flux and
    # flux recovered by the sum

    measurement_array = np.empty(shape=(xpixs, ypixs))

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
            #print 'FILTER: %s' % cube.filter
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
            noise_from_hist = noise_from_histogram(wave_array,
                                                   spaxel_spec,
                                                   redshift,
                                                   cube.filter)

            spaxel_noise = noise[:, i, j]/spaxel_median_flux

            # first search for the linepeak, which may be different
            # to that specified by the systemic redshift
            # set the upper and lower ranges for the t_index search

            t_index = np.argmax(spaxel_spec[line_idx - lower_t:
                                            line_idx + upper_t])

            # need this to be an absolute index
            t_index = t_index + line_idx - lower_t

            # then sum the flux inside the region over which the line
            # will be. Width of line is roughly 0.003, which is 10
            # spectral elements in K and 6 in HK

            lower_limit = t_index - range_lower
            upper_limit = t_index + range_upper

            # take the line counts as the sum over the relevant lambda range

            line_counts = np.nansum(spaxel_spec[lower_limit:
                                                upper_limit])

            noise_from_masked_method = noise_from_masked_spectrum(wave_array,
                                                                  spaxel_spec,
                                                                  redshift,
                                                                  len(spaxel_spec[lower_limit:upper_limit]),
                                                                  line_idx,
                                                                  cube.filter)

            line_counts = line_counts * cube.dL

            # do the gaussian fitting

            plt.close('all')

            try:


                gauss_values, covar = gauss_fit(wave_array[lower_limit: upper_limit],
                                                spaxel_spec[lower_limit: upper_limit],
                                                weights_array_norm[lower_limit: upper_limit],
                                                central_wl)

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
                line_p_noise = line_noise/spaxel_median_flux

            elif noise_method == 'cube':

                sigma_array = spaxel_noise[lower_limit:upper_limit]

                sigma_squared = sigma_array * sigma_array

                line_noise = np.sqrt(np.nansum(sigma_squared))

                line_p_noise = np.std(sigma_array)

            else:

                print 'Please Provide valid noise method'

                raise ValueError('Please provide valid noise method')

            # print 'NOISE COMPARISON %s %s %s %s' % (line_noise,line_p_noise,noise_from_hist,noise_from_masked_method)

            # find the noise reduction factors of the binning methods
            # these feed into the binning_three and binning_five
            # methods to figure out what the new noise should be

            t_red = compute_noise_reduction_factor_three(data,
                                                         xpixs,
                                                         ypixs,
                                                         lower_limit,
                                                         upper_limit)

            f_red = compute_noise_reduction_factor_five(data,
                                                        xpixs,
                                                        ypixs,
                                                        lower_limit,
                                                        upper_limit)

            #print 'REDUCTION FACTORS %s %s' % (t_red,f_red)

            # this must also be multiplied by the spectral resolution
            # print 'This is the original line noise: %s' % line_p_noise

            line_noise = line_noise * cube.dL
            noise_from_masked_method = noise_from_masked_method * cube.dL

#                print 'THIS IS THE SIGNAL %s' % line_counts
#                print 'THIS IS THE NOISE %s' % line_noise

            # be careful with how the signal array is populated

            if np.isnan(line_counts):

                signal_array[i, j] = 0

            else:

                signal_array[i, j] = line_counts*spaxel_median_flux

            # populate the noise array

            noise_array[i, j] = line_noise*spaxel_median_flux

            # compute the signal to noise on the basis of the
            # above calculations

            line_sn = line_counts / noise_from_masked_method

            #print 'THIS IS THE SIGNAL TO NOISE: %s %s %s' % (line_sn,line_counts,line_noise)

            # searching the computed signal to noise in this section

            if np.isnan(line_sn) or np.isinf(line_sn) or np.isclose(line_sn, 0, atol=1E-5):

                # print 'getting rid of nan'

                # we've got a nan entry - get rid of it

                sn_array[i, j] = np.nan
                vel_array[i, j] = np.nan
                disp_array[i, j] = np.nan
                flux_array[i, j] = np.nan
                vel_error_array[i, j] = np.nan
                sig_error_array[i, j] = np.nan

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
                    new_flux = perturb_value(noise_from_hist,
                                             spaxel_spec[lower_limit:
                                                         upper_limit])

                    # fit the gaussian to recover the parameters
                    gauss_values, covar = gauss_fit(wave_array[lower_limit:
                                                               upper_limit],
                                                    new_flux,
                                                    weights_array_norm[lower_limit:upper_limit],
                                                    central_wl)

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

                sn_array[i, j] = line_sn
                vel_array[i, j] = vel_gauss_values['center']

                # sometimes getting bung values for the width of
                # the emission lines

                if sig_gauss_values['center'] > 0 and sig_gauss_values['center'] < 1000:

                    disp_array[i, j] = sig_gauss_values['center']

                else:

                    disp_array[i, j] = np.nan

                flux_array[i, j] = amp_gauss_values['center']*spaxel_median_flux
                vel_error_array[i, j] = vel_gauss_values['sigma']
                sig_error_array[i, j] = sig_gauss_values['sigma']

            # don't bother expanding area if line_sn starts negative

            elif line_sn < 0:

                # print 'Found negative signal %s %s' % (i, j)

                sn_array[i, j] = np.nan
                vel_array[i, j] = np.nan
                disp_array[i, j] = np.nan
                flux_array[i, j] = np.nan
                vel_error_array[i, j] = np.nan
                sig_error_array[i, j] = np.nan

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
                new_noise_from_histogram = noise_from_histogram(wave_array,
                                                                spec,
                                                                redshift,
                                                                cube.filter)
                new_noise_from_masked_method = noise_from_masked_spectrum(wave_array,
                                                                  spec,
                                                                  redshift,
                                                                  len(spec[lower_limit:upper_limit]),
                                                                  line_idx,
                                                                  cube.filter)
                new_noise_from_masked_method = new_noise_from_masked_method * cube.dL
                spec = spec[lower_limit:upper_limit]

                # now that spec has been computed, look at whether
                # the signal to noise of the stack has improved

                new_line_counts = np.nansum(spec)

                new_line_counts = new_line_counts * cube.dL

                new_sn = new_line_counts / new_noise_from_masked_method

                # have to fit gaussian at this point as well
                # and examine similarity between the gaussian fit
                # and the line_counts
                plt.close('all')

                gauss_values, covar = gauss_fit(wave_array[lower_limit: upper_limit],
                                                spec,
                                                weights_array_norm[lower_limit:upper_limit],
                                                central_wl)

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
                        new_flux = perturb_value(new_noise_from_histogram,
                                                 spec)

                        # fit the gaussian to recover the parameters
                        gauss_values, covar = gauss_fit(wave_array[lower_limit:
                                                                   upper_limit],
                                                        new_flux,
                                                        weights_array_norm[lower_limit:upper_limit],
                                                        central_wl)

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
                    sn_array[i, j] = line_sn
                    vel_array[i, j] = vel_gauss_values['center']

                    if sig_gauss_values['center'] > 0 and sig_gauss_values['center'] < 1000:

                        disp_array[i, j] = sig_gauss_values['center']

                    else:

                        disp_array[i, j] = np.nan

                    flux_array[i, j] = amp_gauss_values['center']*spaxel_median_flux
                    vel_error_array[i, j] = vel_gauss_values['sigma']
                    sig_error_array[i, j] = sig_gauss_values['sigma']

                # don't bother expanding area if line_sn starts negative

                elif new_sn < 0:

                    # print 'Found negative signal %s %s' % (i, j)

                    sn_array[i, j] = np.nan
                    vel_array[i, j] = np.nan
                    disp_array[i, j] = np.nan
                    flux_array[i, j] = np.nan
                    vel_error_array[i, j] = np.nan
                    sig_error_array[i, j] = np.nan

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
                    final_noise_from_histogram = noise_from_histogram(wave_array,
                                                                      spec,
                                                                      redshift,
                                                                      cube.filter)
                    final_noise_from_masked_method = noise_from_masked_spectrum(wave_array,
                                                                  spec,
                                                                  redshift,
                                                                  len(spec[lower_limit:upper_limit]),
                                                                  line_idx,
                                                                  cube.filter)
                    final_noise_from_masked_method = final_noise_from_masked_method * cube.dL
                    spec = spec[lower_limit:upper_limit]

                # now that spec has been computed, look at whether
                # the signal to noise of the stack has improved

                    final_line_counts = np.nansum(spec)

                    final_line_counts = cube.dL * final_line_counts

                    final_sn = final_line_counts / final_noise_from_masked_method

                    plt.close('all')

                    gauss_values, covar = gauss_fit(wave_array[lower_limit: upper_limit],
                                                    spec,
                                                    weights_array_norm[lower_limit:upper_limit],
                                                    central_wl)

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

                        sn_array[i, j] = final_sn

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
                            new_flux = perturb_value(final_noise_from_histogram,
                                                     spec)

                            # print new_flux

                            # fit the gaussian to recover the parameters
                            gauss_values, covar = gauss_fit(wave_array[lower_limit:
                                                                       upper_limit],
                                                            new_flux,
                                                            weights_array_norm[lower_limit:upper_limit],
                                                            central_wl)

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

                        sn_array[i, j] = line_sn
                        vel_array[i, j] = vel_gauss_values['center']

                        if sig_gauss_values['center'] > 0 and sig_gauss_values['center'] < 1000:

                            disp_array[i, j] = sig_gauss_values['center']

                        else:

                            disp_array[i, j] = np.nan

                        flux_array[i, j] = amp_gauss_values['center']*spaxel_median_flux
                        vel_error_array[i, j] = vel_gauss_values['sigma']
                        sig_error_array[i, j] = sig_gauss_values['sigma']

                    elif (final_sn > 0 and final_sn < threshold) or \
                         (final_sn > threshold and (int_ratio < g_c_min or int_ratio > g_c_max)) or \
                         (final_sn > threshold and (amp_err > tol or sig_err > tol or cen_err > tol)):

                        # print 'Threshold reached but sum and gauss too disimilar'
                    
                        sn_array[i, j] = np.nan
                        vel_array[i, j] = np.nan
                        disp_array[i, j] = np.nan
                        flux_array[i, j] = np.nan
                        vel_error_array[i, j] = np.nan
                        sig_error_array[i, j] = np.nan

                    else:

                        # didn't reach target - store as nan

                        # print 'no improvement, stop trying to fix'

                        sn_array[i, j] = np.nan
                        vel_array[i, j] = np.nan
                        disp_array[i, j] = np.nan
                        flux_array[i, j] = np.nan
                        vel_error_array[i, j] = np.nan
                        sig_error_array[i, j] = np.nan

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
                                            [5.0, 95.0])
        sig_min, sig_max = np.nanpercentile(disp_array[mask_x_lower:mask_x_upper,
                                                       mask_y_lower:mask_y_upper],
                                            [5.0, 95.0])
        flux_min, flux_max = np.nanpercentile(flux_array[mask_x_lower:mask_x_upper,
                                                         mask_y_lower:mask_y_upper],
                                              [5.0, 95.0])

        s_min, s_max = np.nanpercentile(signal_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [5.0, 95.0])

        sn_min, sn_max = np.nanpercentile(sn_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [5.0, 95.0])

        g_min, g_max = np.nanpercentile(measurement_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [5.0, 95.0])

        er_min, er_max = np.nanpercentile(vel_error_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [5.0, 95.0])

        sig_er_min, sig_er_max = np.nanpercentile(sig_error_array[mask_x_lower:mask_x_upper,
                                                     mask_y_lower:mask_y_upper],
                                        [5.0, 95.0])

    except TypeError:

        # origin of the error is lack of good S/N data
        # can set the max and min at whatever
        vel_min, vel_max = [-100, 100]
        sig_min, sig_max = [0, 150]
        flux_min, flux_max = [0, 5E-3]
        s_min, s_max = [0, 0.01]
        sn_min, sn_max = [0, 10]
        g_min, g_max = [0, 1.5]
        er_min, er_max = [0, 100]
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
    ax[0].set_title('[OIII] Flux')

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
                                                    [5.0, 95.0])
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
    ax[1].set_title('[OIII] Velocity')

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
    ax[2].set_title('[OIII] Dispersion')

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

    # return the noise, signal and flux arrays for potential
    # voronoi binning
    
    return [noise_array[mask_x_lower:mask_x_upper,
                        mask_y_lower:mask_y_upper],
            signal_array[mask_x_lower:mask_x_upper,
                         mask_y_lower:mask_y_upper],
            flux_array,
            masked_vel_array,
            masked_int_sig_array]

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
        p_noise_values.append(np.nanstd(entry[lower_l:upper_l]))

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

    final_noise = np.nanstd(noise_values)
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
              central_wl):

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

    # perform the fit
    out = gmod.fit(fit_flux,
                   pars,
                   x=fit_wl)

    # print the fit report
    # print out.fit_report()

    # plot to make sure things are working
#    fig, ax = plt.subplots(figsize=(14, 6))
#    ax.plot(fit_wl, fit_flux, color='blue')
#    ax.plot(fit_wl, out.best_fit, color='red')
#    plt.show()
#    plt.close('all')

    return out.best_values, out.covar

def noise_from_masked_spectrum(wave_array,
                               spec,
                               redshift,
                               region_length,
                               spec_index,
                               filt):
    
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
                                                            filt)

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
                         filt):
    
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
                                                            filt)

    # plot

#    fig, ax = plt.subplots(1, 1, figsize=(16,10))
#    ax.plot(wave_array,sky_and_emission_line_masked_spec)
#    plt.show()
#    plt.close('all')

    # recover the compressed spectrum

    compressed_spec = sky_and_emission_line_masked_spec[100:1700].compressed()

    # take only the 5th to 95th percentile
    compressed_spec = np.sort(compressed_spec)
    compressed_spec = compressed_spec[int(len(compressed_spec)/20.0):int(0.95*len(compressed_spec))]

    bins, centres = np.histogram(compressed_spec, bins=20)
    noise_mod = GaussianModel()
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
    return noise


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
                            filt):

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
                                                   filt)
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
                        filt):
    
    # choose the appropriate sky dictionary for the filter
    if filt == 'YJ':

        oii_rest = 0.37284835
        oii_shifted = (1 + redshift) * oii_rest
        oii_shifted_index = np.argmin(np.abs(wave_array - oii_shifted))
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
        h_beta_shifted_index = np.argmin(np.abs(wave_array - h_beta_shifted))
        h_beta_shifted_min = wave_array[h_beta_shifted_index-20]
        h_beta_shifted_max = wave_array[h_beta_shifted_index+20]
        oiii_4960_rest = 0.4960295
        oiii_4960_shifted = (1 + redshift) * oiii_4960_rest
        oiii_4960_shifted_index = np.argmin(np.abs(wave_array - oiii_4960_shifted))
        oiii_4960_shifted_min = wave_array[oiii_4960_shifted_index-20]
        oiii_4960_shifted_max = wave_array[oiii_4960_shifted_index+20]
        oiii_5008_rest = 0.5008239
        oiii_5008_shifted = (1 + redshift) * oiii_5008_rest
        oiii_5008_shifted_index = np.argmin(np.abs(wave_array - oiii_5008_shifted))
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
        h_alpha_shifted_index = np.argmin(np.abs(wave_array - h_alpha_shifted))
        h_alpha_shifted_min = wave_array[h_alpha_shifted_index-20]
        h_alpha_shifted_max = wave_array[h_alpha_shifted_index+20]
        nii_rest = 0.658523
        nii_shifted = (1 + redshift) * nii_rest
        nii_shifted_index = np.argmin(np.abs(wave_array - nii_shifted))
        nii_shifted_min = wave_array[nii_shifted_index-20]
        nii_shifted_max = wave_array[nii_shifted_index+20]
        sii_lower_rest = 0.671829
        sii_lower_shifted = (1 + redshift) * sii_lower_rest
        sii_lower_shifted_index = np.argmin(np.abs(wave_array - sii_lower_shifted))
        sii_lower_shifted_min = wave_array[sii_lower_shifted_index-20]
        sii_lower_shifted_max = wave_array[sii_lower_shifted_index+20]
        sii_upper_rest = 0.671829
        sii_upper_shifted = (1 + redshift) * sii_upper_rest
        sii_upper_shifted_index = np.argmin(np.abs(wave_array - sii_upper_shifted))
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

multi_vel_field_stott('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Kband/KLP_K_NAMES.txt',
                      'ha',
                      3.0,
                      g_c_min=0.5,
                      g_c_max=1.5,
                      method='median')