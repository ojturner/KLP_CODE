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

# create a single pdf/image with all of the line fits on it. 
# makes it much easier to diagnose what we can do with these data.

def multi_vel_field_all_lines(infile_k,
                              infile_h,
                              infile_yj,
                              threshold,
                              g_c_min,
                              g_c_max,
                              ntimes=200,
                              spatial_smooth=False,
                              spectral_smooth=False,
                              smoothing_psf=0.3,
                              spectral_smooth_width=2,
                              prog='klp',
                              **kwargs):
    
    noise_method = 'mask'
    table_k = ascii.read(infile_k)
    table_h = ascii.read(infile_h)
    table_yj = ascii.read(infile_yj)
    redshift_k = table_k['redshift']
    ra_k = table_k['RA']
    dec_k = table_k['DEC']

    # assign variables to all the essential properties
    # in each of the wavebands

    obj_name_k = table_k['Filename']
    sky_cube_k = table_k['sky_cube']
    central_x_k = table_k['Central_x']
    central_y_k = table_k['Central_y']
    mask_x_lower_k = table_k['mask_x_lower']
    mask_x_upper_k = table_k['mask_x_upper']
    mask_y_lower_k = table_k['mask_y_lower']
    mask_y_upper_k = table_k['mask_y_upper']
    image_type =  table_k['im_type']

    obj_name_h = table_h['Filename']
    sky_cube_h = table_h['sky_cube']
    central_x_h = table_h['Central_x']
    central_y_h = table_h['Central_y']
    mask_x_lower_h = table_h['mask_x_lower']
    mask_x_upper_h = table_h['mask_x_upper']
    mask_y_lower_h = table_h['mask_y_lower']
    mask_y_upper_h = table_h['mask_y_upper']

    obj_name_yj = table_yj['Filename']
    sky_cube_yj = table_yj['sky_cube']
    central_x_yj = table_yj['Central_x']
    central_y_yj = table_yj['Central_y']
    mask_x_lower_yj = table_yj['mask_x_lower']
    mask_x_upper_yj = table_yj['mask_x_upper']
    mask_y_lower_yj = table_yj['mask_y_lower']
    mask_y_upper_yj = table_yj['mask_y_upper']

    for obj_name__k, sky_cube__k, central_x__k, central_y__k, mask_x_lower__k, mask_x_upper__k, \
        mask_y_lower__k, mask_y_upper__k, obj_name__h, sky_cube__h, central_x__h, central_y__h, \
        mask_x_lower__h, mask_x_upper__h, mask_y_lower__h, mask_y_upper__h, \
        obj_name__yj, sky_cube__yj, central_x__yj, central_y__yj, mask_x_lower__yj, mask_x_upper__yj, \
        mask_y_lower__yj, mask_y_upper__yj, im_type, redshift, ra, dec in zip(obj_name_k,
                                                                              sky_cube_k,
                                                                              central_x_k,
                                                                              central_x_h,
                                                                              mask_x_lower_k,
                                                                              mask_x_upper_k,
                                                                              mask_y_lower_k,
                                                                              mask_y_upper_k,
                                                                              obj_name_h,
                                                                              sky_cube_h,
                                                                              central_x_h,
                                                                              central_y_k,
                                                                              mask_x_lower_h,
                                                                              mask_x_upper_h,
                                                                              mask_y_lower_h,
                                                                              mask_y_upper_h,
                                                                              obj_name_yj,
                                                                              sky_cube_yj,
                                                                              central_x_yj,
                                                                              central_y_yj,
                                                                              mask_x_lower_yj,
                                                                              mask_x_upper_yj,
                                                                              mask_y_lower_yj,
                                                                              mask_y_upper_yj,
                                                                              image_type,
                                                                              redshift_k,
                                                                              ra_k,
                                                                              dec_k):
        
        redshift = float(redshift)
        
        # fit the line profiles in the datacubes in all the bands and make
        # a figure to accommodate the resultant data

        # define the science directory for each cube
        sci_dir = obj_name__k[:len(obj_name__k) - obj_name__k[::-1].find("/") - 1]

        gal_name = obj_name__k[len(obj_name__k) - obj_name__k[::-1].find("/"):]
        gal_name = gal_name[26:-5]

        print "\nDoing %s (redshift = %.3f) ..." % (obj_name__k, redshift)

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

        # first extract the HST images for that galaxy
        # there are 6 different wavebands from V-H covered in GS
        # and 4 in COSMOS so can almost fill all the spaces
        # fill in now the different wavebands  

        if im_type == 'GS':
            image_file_1 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS/F606W/hlsp_hlf_hst_acs-60mas_goodss_f606w_v1.5_sci.fits')
            image_file_2 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS/F814W/hlsp_hlf_hst_acs-60mas_goodss_f814w_v1.5_sci.fits')
            image_file_3 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS/F850LP/hlsp_hlf_hst_acs-60mas_goodss_f850lp_v1.5_sci.fits')
            image_file_4 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS/F105W/hlsp_hlf_hst_wfc3-60mas_goodss_f105w_v1.5_sci.fits')
            image_file_5 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS/F125W/hlsp_hlf_hst_wfc3-60mas_goodss_f125w_v1.5_sci.fits')
            image_file_6 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS/F160W/hlsp_hlf_hst_wfc3-60mas_goodss_f160w_v1.5_sci.fits')
        elif im_type == 'COS':
            image_file_1 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_COSMOS/F606W/hlsp_candels_hst_acs_cos-tot_f606w_v1.0_drz.fits')
            image_file_2 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_COSMOS/F814W/hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits')
            image_file_3 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_COSMOS/F814W/hlsp_candels_hst_acs_cos-tot_f814w_v1.0_drz.fits')
            image_file_4 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_COSMOS/F125W/hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits')
            image_file_5 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_COSMOS/F125W/hlsp_candels_hst_wfc3_cos-tot_f125w_v1.0_drz.fits')
            image_file_6 = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_COSMOS/F160W/hlsp_candels_hst_wfc3_cos-tot_f160w_v1.0_drz.fits')

        # work the extraction from HST image magic for each of the
        # different wavebands

        # F606
        image_header_1 = image_file_1[0].header
        image_data_1 = image_file_1[0].data
        w = WCS(image_header_1)
        lon, lat = w.wcs_world2pix(ra, dec, 1)
        lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
        lon_unit = int(np.round((3.0 / (lon_scale * 3600)) / 2.0))
        lat_unit = int(np.round((3.0 / (lat_scale * 3600)) / 2.0))
        # and get the extraction stamps for sci and wht, ready to save
        extraction_stamp_1 = image_data_1[lat - lat_unit:lat + lat_unit,
                                          lon - lon_unit:lon + lon_unit]
        # F814W
        image_header_2 = image_file_2[0].header
        image_data_2 = image_file_2[0].data
        w = WCS(image_header_2)
        lon, lat = w.wcs_world2pix(ra, dec, 1)
        lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
        lon_unit = int(np.round((3.0 / (lon_scale * 3600)) / 2.0))
        lat_unit = int(np.round((3.0 / (lat_scale * 3600)) / 2.0))
        # and get the extraction stamps for sci and wht, ready to save
        extraction_stamp_2 = image_data_2[lat - lat_unit:lat + lat_unit,
                                          lon - lon_unit:lon + lon_unit]

        # F850LP
        image_header_3 = image_file_3[0].header
        image_data_3 = image_file_3[0].data
        w = WCS(image_header_3)
        lon, lat = w.wcs_world2pix(ra, dec, 1)
        lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
        lon_unit = int(np.round((3.0 / (lon_scale * 3600)) / 2.0))
        lat_unit = int(np.round((3.0 / (lat_scale * 3600)) / 2.0))
        # and get the extraction stamps for sci and wht, ready to save
        extraction_stamp_3 = image_data_3[lat - lat_unit:lat + lat_unit,
                                          lon - lon_unit:lon + lon_unit]
        # F105W
        image_header_4 = image_file_4[0].header
        image_data_4 = image_file_4[0].data
        w = WCS(image_header_4)
        lon, lat = w.wcs_world2pix(ra, dec, 1)
        lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
        lon_unit = int(np.round((3.0 / (lon_scale * 3600)) / 2.0))
        lat_unit = int(np.round((3.0 / (lat_scale * 3600)) / 2.0))
        # and get the extraction stamps for sci and wht, ready to save
        extraction_stamp_4 = image_data_4[lat - lat_unit:lat + lat_unit,
                                          lon - lon_unit:lon + lon_unit]
        # F125W
        image_header_5 = image_file_5[0].header
        image_data_5 = image_file_5[0].data
        w = WCS(image_header_5)
        lon, lat = w.wcs_world2pix(ra, dec, 1)
        lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
        lon_unit = int(np.round((3.0 / (lon_scale * 3600)) / 2.0))
        lat_unit = int(np.round((3.0 / (lat_scale * 3600)) / 2.0))
        # and get the extraction stamps for sci and wht, ready to save
        extraction_stamp_5 = image_data_5[lat - lat_unit:lat + lat_unit,
                                          lon - lon_unit:lon + lon_unit]
        # F160W
        image_header_6 = image_file_6[0].header
        image_data_6 = image_file_6[0].data
        w = WCS(image_header_6)
        lon, lat = w.wcs_world2pix(ra, dec, 1)
        lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
        lon_unit = int(np.round((3.0 / (lon_scale * 3600)) / 2.0))
        lat_unit = int(np.round((3.0 / (lat_scale * 3600)) / 2.0))
        # and get the extraction stamps for sci and wht, ready to save
        extraction_stamp_6 = image_data_6[lat - lat_unit:lat + lat_unit,
                                          lon - lon_unit:lon + lon_unit]

        # get the integrated spectrum for each of the emission lines

        # oii
        oii_int_spec, oii_int_wave, oii_cont, oii_cont_vmin, oii_cont_vmax, masks_im = int_spec_fit.integrated_spec_extract(obj_name__yj,
                                                                          central_x__yj,
                                                                          central_y__yj,
                                                                          redshift,
                                                                          'oii',
                                                                          prog=prog)

        # hb
        hb_int_spec, hb_int_wave, hb_cont, hb_cont_vmin, hb_cont_vmax, masks_im = int_spec_fit.integrated_spec_extract(obj_name__h,
                                                                          central_x__h,
                                                                          central_y__h,
                                                                          redshift,
                                                                          'hb',
                                                                          prog=prog)

        # oiiiweak
        oiiiweak_int_spec, oiiiweak_int_wave, oiiiweak_cont, oiiiweak_cont_vmin, oiiiweak_cont_vmax, masks_im = int_spec_fit.integrated_spec_extract(obj_name__h,
                                                                          central_x__h,
                                                                          central_y__h,
                                                                          redshift,
                                                                          'oiiiweak',
                                                                          prog=prog)

        # oiii
        oiii_int_spec, oiii_int_wave, oiii_cont, oiii_cont_vmin, oiii_cont_vmax, masks_im = int_spec_fit.integrated_spec_extract(obj_name__h,
                                                                          central_x__h,
                                                                          central_y__h,
                                                                          redshift,
                                                                          'oiii',
                                                                          prog=prog)

        # ha
        ha_int_spec, ha_int_wave, ha_cont, ha_cont_vmin, ha_cont_vmax, masks_im = int_spec_fit.integrated_spec_extract(obj_name__k,
                                                                          central_x__k,
                                                                          central_y__k,
                                                                          redshift,
                                                                          'ha',
                                                                          prog=prog)

        # nii
        nii_int_spec, nii_int_wave, nii_cont, nii_cont_vmin, nii_cont_vmax, masks_im = int_spec_fit.integrated_spec_extract(obj_name__k,
                                                                          central_x__k,
                                                                          central_y__k,
                                                                          redshift,
                                                                          'nii',
                                                                          prog=prog)                           

        # one at a time calculate the line maps 
        # and then plot
        oii_grid = spaxel_fit.vel_field_stott_binning(obj_name__yj,
                                                      sky_cube__yj,
                                                      'oii',
                                                      redshift,
                                                      threshold,
                                                      mask_x_lower__yj,
                                                      mask_x_upper__yj,
                                                      mask_y_lower__yj,
                                                      mask_y_upper__yj,
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
                                                      prog=prog)
        hb_grid = spaxel_fit.vel_field_stott_binning(obj_name__h,
                                                      sky_cube__h,
                                                      'hb',
                                                      redshift,
                                                      threshold,
                                                      mask_x_lower__h,
                                                      mask_x_upper__h,
                                                      mask_y_lower__h,
                                                      mask_y_upper__h,
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
                                                      prog=prog)
        oiiiweak_grid = spaxel_fit.vel_field_stott_binning(obj_name__h,
                                                      sky_cube__h,
                                                      'oiiiweak',
                                                      redshift,
                                                      threshold,
                                                      mask_x_lower__h,
                                                      mask_x_upper__h,
                                                      mask_y_lower__h,
                                                      mask_y_upper__h,
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
                                                      prog=prog)
        oiii_grid = spaxel_fit.vel_field_stott_binning(obj_name__h,
                                                      sky_cube__h,
                                                      'oiii',
                                                      redshift,
                                                      threshold,
                                                      mask_x_lower__h,
                                                      mask_x_upper__h,
                                                      mask_y_lower__h,
                                                      mask_y_upper__h,
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
                                                      prog=prog)
        ha_grid = spaxel_fit.vel_field_stott_binning(obj_name__k,
                                                      sky_cube__k,
                                                      'ha',
                                                      redshift,
                                                      threshold,
                                                      mask_x_lower__k,
                                                      mask_x_upper__k,
                                                      mask_y_lower__k,
                                                      mask_y_upper__k,
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
                                                      prog=prog)
        nii_grid = spaxel_fit.vel_field_stott_binning(obj_name__k,
                                                      sky_cube__k,
                                                      'nii',
                                                      redshift,
                                                      threshold,
                                                      mask_x_lower__k,
                                                      mask_x_upper__k,
                                                      mask_y_lower__k,
                                                      mask_y_upper__k,
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
                                                      prog=prog)

        # return the sky lines for plotting over the integrated spectrum
        yj_sky_dict = mask_the_sky.ret_yj_sky()
        h_sky_dict = mask_the_sky.ret_h_sky()
        k_sky_dict = mask_the_sky.ret_k_sky()


        # define the figure which will accommodate all of the data
        fig, axes = plt.subplots(6, 8, figsize=(40, 32), gridspec_kw= {'width_ratios':[1,1,1,1,1,1,1,3]})
        import string
        fig.suptitle(gal_name,
                     fontsize=18,
                     fontweight='bold')

#        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#        rc('font', weight='bold')
#        rc('text', usetex=True)
#        rc('axes', linewidth=2)
#        plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

        # populate the big array with the oii data
        # first must set the limits for the colourbars

        try:
            vel_min, vel_max = np.nanpercentile(oii_grid[1],[15.0, 85.0])
        except TypeError:
            vel_min, vel_max = [-100, 100]
        try:
            sig_min, sig_max = np.nanpercentile(oii_grid[2],[15.0, 85.0])
        except TypeError:
            sig_min, sig_max = [0, 150]
        try:
            flux_min, flux_max = np.nanpercentile(oii_grid[0],[15.0, 85.0])
        except TypeError:
            flux_min, flux_max = [1E-20, 5E-18]
        try:
            s_min, s_max = np.nanpercentile(oii_grid[3],[15.0, 85.0])
        except TypeError:
            s_min, s_max = [1E-20, 5E-18]
        try:
            noise_min, noise_max = np.nanpercentile(oii_grid[4],[15.0, 85.0])
        except TypeError:
            noise_min, noise_max = [1E-20, 5E-18]
        try:
            sn_min, sn_max = np.nanpercentile(oii_grid[5],[15.0, 85.0])
        except TypeError:
            sn_min, sn_max = [0, 10]
        try:
            g_min, g_max = np.nanpercentile(oii_grid[6],[15.0, 85.0])
        except TypeError:
            g_min, g_max = [0, 1.5]
        try:
            er_min, er_max = np.nanpercentile(oii_grid[7],[15.0, 85.0])
        except TypeError:
            er_min, er_max = [0, 100]
        try:
            sig_er_min, sig_er_max = np.nanpercentile(oii_grid[8],[15.0, 85.0])
        except TypeError:
            sig_er_min, sig_er_max = [0, 100]

        # get rid of pesky tick labels
        for entry in axes:
            for ax in entry:
                ax.set_xticks([])
                ax.set_yticks([])

        # label each row with the appropriate emission line
        axes[0][1].set_ylabel('[OII]3727',
                              fontsize=18,
                              fontweight='bold')
        axes[1][1].set_ylabel('Hb',
                              fontsize=18,
                              fontweight='bold')
        axes[2][1].set_ylabel('[OIII]4959',
                              fontsize=18,
                              fontweight='bold')
        axes[3][1].set_ylabel('[OIII]5007',
                              fontsize=18,
                              fontweight='bold')
        axes[4][1].set_ylabel('Ha',
                              fontsize=18,
                              fontweight='bold')
        axes[5][1].set_ylabel('[NII]6585',
                              fontsize=18,
                              fontweight='bold')

        # now populate the plot with all of these aspects

        # set the title
        axes[0][0].set_title('HST F606W',
                             fontsize=15,
                             fontweight='bold')
        # extraction stamp
        axes[0][0].imshow(extraction_stamp_1,
                          interpolation='nearest',
                          cmap=plt.cm.nipy_spectral)

        # flux
        im = axes[0][1].imshow(oii_grid[0],
                               cmap=plt.get_cmap('jet'),
                               vmin=flux_min,
                               vmax=flux_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[0][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        axes[0][1].set_title('Flux',
                             fontsize=15,
                             fontweight='bold')

        # velocity
        im = axes[0][2].imshow(oii_grid[1],
                               cmap=plt.get_cmap('jet'),
                               vmin=vel_min,
                               vmax=vel_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[0][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        axes[0][2].set_title('Velocity',
                             fontsize=15,
                             fontweight='bold')

        # dispersion
        im = axes[0][3].imshow(oii_grid[2],
                               cmap=plt.get_cmap('jet'),
                               vmin=sig_min,
                               vmax=sig_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[0][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        axes[0][3].set_title('Dispersion',
                             fontsize=15,
                             fontweight='bold')

        # signal
        im = axes[0][4].imshow(oii_grid[3],
                               cmap=plt.get_cmap('jet'),
                               vmin=s_min,
                               vmax=s_max,
                               interpolation='nearest')
        # plot also the extraction aperture
        axes[0][4].contour(masks_im[0],
                           masks_im[1],
                           masks_im[2],
                           1,
                           linewidths=3)

        # add colourbar to each plot
        divider = make_axes_locatable(axes[0][4])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        axes[0][4].set_title('Narrowband',
                             fontsize=15,
                             fontweight='bold')

        # CONTINUUM
        im = axes[0][5].imshow(oii_cont,
                               cmap=plt.cm.nipy_spectral,
                               vmin=oii_cont_vmin,
                               vmax=oii_cont_vmax,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[0][5])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        axes[0][5].set_title('Continuum',
                             fontsize=15,
                             fontweight='bold')

        # signal to noise
        im = axes[0][6].imshow(oii_grid[5],
                               cmap=plt.get_cmap('jet'),
                               vmin=sn_min,
                               vmax=sn_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[0][6])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # set the title
        axes[0][6].set_title('Signal to noise',
                             fontsize=15,
                             fontweight='bold')

        # sum to gauss
#        im = axes[0][7].imshow(oii_grid[6],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=g_min,
#                               vmax=g_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[0][7])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # set the title
#        axes[0][7].set_title('Sum to Gauss',
#                             fontsize=15,
#                             fontweight='bold')
#        # velocity error
#        im = axes[0][8].imshow(oii_grid[7],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=er_min,
#                               vmax=er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[0][8])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # set the title
#        axes[0][8].set_title('Vel error',
#                             fontsize=15,
#                             fontweight='bold')
#        # sigma error
#        im = axes[0][9].imshow(oii_grid[8],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=sig_er_min,
#                               vmax=sig_er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[0][9])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # set the title
#        axes[0][9].set_title('Sig error',
#                             fontsize=15,
#                             fontweight='bold')

        # now plotting the apperture spectrum in 8th position instead

        axes[0][7].plot(oii_int_wave,
                        oii_int_spec,
                        drawstyle='steps-mid',
                        lw=2)
        axes[0][7].axhline(y=0E-19,
                           xmin=0,
                           xmax=1,
                           color='black',
                           ls='--',
                           lw=2)
        for ranges in yj_sky_dict.values():
            axes[0][7].axvspan(ranges[0],
                               ranges[1],
                               alpha=0.5,
                               color='grey')
        axes[0][7].set_xlim(oii_int_wave[0],
                            oii_int_wave[-1])
        # set the title
        axes[0][7].set_title('Apperture Spec',
                             fontsize=15,
                             fontweight='bold')

        try:
            vel_min, vel_max = np.nanpercentile(hb_grid[1],[15.0, 85.0])
        except TypeError:
            vel_min, vel_max = [-100, 100]
        try:
            sig_min, sig_max = np.nanpercentile(hb_grid[2],[15.0, 85.0])
        except TypeError:
            sig_min, sig_max = [0, 150]
        try:
            flux_min, flux_max = np.nanpercentile(hb_grid[0],[15.0, 85.0])
        except TypeError:
            flux_min, flux_max = [1E-20, 5E-18]
        try:
            s_min, s_max = np.nanpercentile(hb_grid[3],[15.0, 85.0])
        except TypeError:
            s_min, s_max = [1E-20, 5E-18]
        try:
            noise_min, noise_max = np.nanpercentile(hb_grid[4],[15.0, 85.0])
        except TypeError:
            noise_min, noise_max = [1E-20, 5E-18]
        try:
            sn_min, sn_max = np.nanpercentile(hb_grid[5],[15.0, 85.0])
        except TypeError:
            sn_min, sn_max = [0, 10]
        try:
            g_min, g_max = np.nanpercentile(hb_grid[6],[15.0, 85.0])
        except TypeError:
            g_min, g_max = [0, 1.5]
        try:
            er_min, er_max = np.nanpercentile(hb_grid[7],[15.0, 85.0])
        except TypeError:
            er_min, er_max = [0, 100]
        try:
            sig_er_min, sig_er_max = np.nanpercentile(hb_grid[8],[15.0, 85.0])
        except TypeError:
            sig_er_min, sig_er_max = [0, 100]

        # set the title
        axes[1][0].set_title('HST F814W',
                             fontsize=15,
                             fontweight='bold')

        #  extraction stamp
        axes[1][0].imshow(extraction_stamp_2,
                          interpolation='nearest',
                          cmap=plt.cm.nipy_spectral)

        # flux
        im = axes[1][1].imshow(hb_grid[0],
                               cmap=plt.get_cmap('jet'),
                               vmin=flux_min,
                               vmax=flux_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[1][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # velocity
        im = axes[1][2].imshow(hb_grid[1],
                               cmap=plt.get_cmap('jet'),
                               vmin=vel_min,
                               vmax=vel_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[1][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # dispersion
        im = axes[1][3].imshow(hb_grid[2],
                               cmap=plt.get_cmap('jet'),
                               vmin=sig_min,
                               vmax=sig_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[1][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal
        im = axes[1][4].imshow(hb_grid[3],
                               cmap=plt.get_cmap('jet'),
                               vmin=s_min,
                               vmax=s_max,
                               interpolation='nearest')
        # plot also the extraction aperture
        axes[1][4].contour(masks_im[0],
                           masks_im[1],
                           masks_im[2],
                           1,
                           linewidths=3)

        # add colourbar to each plot
        divider = make_axes_locatable(axes[1][4])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # CONTINUUM
        im = axes[1][5].imshow(hb_cont,
                               cmap=plt.cm.nipy_spectral,
                               vmin=hb_cont_vmin,
                               vmax=hb_cont_vmax,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[1][5])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal to noise
        im = axes[1][6].imshow(hb_grid[5],
                               cmap=plt.get_cmap('jet'),
                               vmin=sn_min,
                               vmax=sn_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[1][6])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # sum to gauss
#        im = axes[1][7].imshow(hb_grid[6],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=g_min,
#                               vmax=g_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[1][7])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # velocity error
#        im = axes[1][8].imshow(hb_grid[7],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=er_min,
#                               vmax=er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[1][8])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # sigma error
#        im = axes[1][9].imshow(hb_grid[8],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=sig_er_min,
#                               vmax=sig_er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[1][9])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)


        # now plotting the apperture spectrum in 8th position instead

        axes[1][7].plot(hb_int_wave,
                        hb_int_spec,
                        drawstyle='steps-mid',
                        lw=2)
        axes[1][7].axhline(y=0E-19,
                           xmin=0,
                           xmax=1,
                           color='black',
                           ls='--',
                           lw=2)
        for ranges in h_sky_dict.values():
            axes[1][7].axvspan(ranges[0],
                               ranges[1],
                               alpha=0.5,
                               color='grey')
        axes[1][7].set_xlim(hb_int_wave[0],
                            hb_int_wave[-1])

        try:
            vel_min, vel_max = np.nanpercentile(oiiiweak_grid[1],[15.0, 85.0])
        except TypeError:
            vel_min, vel_max = [-100, 100]
        try:
            sig_min, sig_max = np.nanpercentile(oiiiweak_grid[2],[15.0, 85.0])
        except TypeError:
            sig_min, sig_max = [0, 150]
        try:
            flux_min, flux_max = np.nanpercentile(oiiiweak_grid[0],[15.0, 85.0])
        except TypeError:
            flux_min, flux_max = [1E-20, 5E-18]
        try:
            s_min, s_max = np.nanpercentile(oiiiweak_grid[3],[15.0, 85.0])
        except TypeError:
            s_min, s_max = [1E-20, 5E-18]
        try:
            noise_min, noise_max = np.nanpercentile(oiiiweak_grid[4],[15.0, 85.0])
        except TypeError:
            noise_min, noise_max = [1E-20, 5E-18]
        try:
            sn_min, sn_max = np.nanpercentile(oiiiweak_grid[5],[15.0, 85.0])
        except TypeError:
            sn_min, sn_max = [0, 10]
        try:
            g_min, g_max = np.nanpercentile(oiiiweak_grid[6],[15.0, 85.0])
        except TypeError:
            g_min, g_max = [0, 1.5]
        try:
            er_min, er_max = np.nanpercentile(oiiiweak_grid[7],[15.0, 85.0])
        except TypeError:
            er_min, er_max = [0, 100]
        try:
            sig_er_min, sig_er_max = np.nanpercentile(oiiiweak_grid[8],[15.0, 85.0])
        except TypeError:
            sig_er_min, sig_er_max = [0, 100]

        # set the title
        axes[2][0].set_title('HST F850LP',
                             fontsize=15,
                             fontweight='bold')

        #  extraction stamp
        axes[2][0].imshow(extraction_stamp_3,
                          interpolation='nearest',
                          cmap=plt.cm.nipy_spectral)

        # flux
        im = axes[2][1].imshow(oiiiweak_grid[0],
                               cmap=plt.get_cmap('jet'),
                               vmin=flux_min,
                               vmax=flux_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[2][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # velocity
        im = axes[2][2].imshow(oiiiweak_grid[1],
                               cmap=plt.get_cmap('jet'),
                               vmin=vel_min,
                               vmax=vel_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[2][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # dispersion
        im = axes[2][3].imshow(oiiiweak_grid[2],
                               cmap=plt.get_cmap('jet'),
                               vmin=sig_min,
                               vmax=sig_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[2][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal
        im = axes[2][4].imshow(oiiiweak_grid[3],
                               cmap=plt.get_cmap('jet'),
                               vmin=s_min,
                               vmax=s_max,
                               interpolation='nearest')
        # plot also the extraction aperture
        axes[2][4].contour(masks_im[0],
                           masks_im[1],
                           masks_im[2],
                           1,
                           linewidths=3)

        # add colourbar to each plot
        divider = make_axes_locatable(axes[2][4])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # CONTINUUM
        im = axes[2][5].imshow(oiiiweak_cont,
                               cmap=plt.cm.nipy_spectral,
                               vmin=oiiiweak_cont_vmin,
                               vmax=oiiiweak_cont_vmax,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[2][5])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal to noise
        im = axes[2][6].imshow(oiiiweak_grid[5],
                               cmap=plt.get_cmap('jet'),
                               vmin=sn_min,
                               vmax=sn_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[2][6])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # sum to gauss
#        im = axes[2][7].imshow(oiiiweak_grid[6],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=g_min,
#                               vmax=g_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[2][7])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # velocity error
#        im = axes[2][8].imshow(oiiiweak_grid[7],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=er_min,
#                               vmax=er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[2][8])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # sigma error
#        im = axes[2][9].imshow(oiiiweak_grid[8],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=sig_er_min,
#                               vmax=sig_er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[2][9])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)


        # now plotting the apperture spectrum in 8th position instead

        axes[2][7].plot(oiiiweak_int_wave,
                        oiiiweak_int_spec,
                        drawstyle='steps-mid',
                        lw=2)

        axes[2][7].axhline(y=0E-19,
                           xmin=0,
                           xmax=1,
                           color='black',
                           ls='--',
                           lw=2)
        for ranges in h_sky_dict.values():
            axes[2][7].axvspan(ranges[0],
                               ranges[1],
                               alpha=0.5,
                               color='grey')
        axes[2][7].set_xlim(oiiiweak_int_wave[0],
                            oiiiweak_int_wave[-1])


        try:
            vel_min, vel_max = np.nanpercentile(oiii_grid[1],[15.0, 85.0])
        except TypeError:
            vel_min, vel_max = [-100, 100]
        try:
            sig_min, sig_max = np.nanpercentile(oiii_grid[2],[15.0, 85.0])
        except TypeError:
            sig_min, sig_max = [0, 150]
        try:
            flux_min, flux_max = np.nanpercentile(oiii_grid[0],[15.0, 85.0])
        except TypeError:
            flux_min, flux_max = [1E-20, 5E-18]
        try:
            s_min, s_max = np.nanpercentile(oiii_grid[3],[15.0, 85.0])
        except TypeError:
            s_min, s_max = [1E-20, 5E-18]
        try:
            noise_min, noise_max = np.nanpercentile(oiii_grid[4],[15.0, 85.0])
        except TypeError:
            noise_min, noise_max = [1E-20, 5E-18]
        try:
            sn_min, sn_max = np.nanpercentile(oiii_grid[5],[15.0, 85.0])
        except TypeError:
            sn_min, sn_max = [0, 10]
        try:
            g_min, g_max = np.nanpercentile(oiii_grid[6],[15.0, 85.0])
        except TypeError:
            g_min, g_max = [0, 1.5]
        try:
            er_min, er_max = np.nanpercentile(oiii_grid[7],[15.0, 85.0])
        except TypeError:
            er_min, er_max = [0, 100]
        try:
            sig_er_min, sig_er_max = np.nanpercentile(oiii_grid[8],[15.0, 85.0])
        except TypeError:
            sig_er_min, sig_er_max = [0, 100]

        # set the title
        axes[3][0].set_title('HST F105W',
                             fontsize=15,
                             fontweight='bold')

        #  extraction stamp
        axes[3][0].imshow(extraction_stamp_4,
                          interpolation='nearest',
                          cmap=plt.cm.nipy_spectral)

        # flux
        im = axes[3][1].imshow(oiii_grid[0],
                               cmap=plt.get_cmap('jet'),
                               vmin=flux_min,
                               vmax=flux_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[3][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # velocity
        im = axes[3][2].imshow(oiii_grid[1],
                               cmap=plt.get_cmap('jet'),
                               vmin=vel_min,
                               vmax=vel_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[3][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # dispersion
        im = axes[3][3].imshow(oiii_grid[2],
                               cmap=plt.get_cmap('jet'),
                               vmin=sig_min,
                               vmax=sig_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[3][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal
        im = axes[3][4].imshow(oiii_grid[3],
                               cmap=plt.get_cmap('jet'),
                               vmin=s_min,
                               vmax=s_max,
                               interpolation='nearest')
        # plot also the extraction aperture
        axes[3][4].contour(masks_im[0],
                           masks_im[1],
                           masks_im[2],
                           1,
                           linewidths=3)

        # add colourbar to each plot
        divider = make_axes_locatable(axes[3][4])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # CONTINUUM
        im = axes[3][5].imshow(oiii_cont,
                               cmap=plt.cm.nipy_spectral,
                               vmin=oiii_cont_vmin,
                               vmax=oiii_cont_vmax,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[3][5])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal to noise
        im = axes[3][6].imshow(oiii_grid[5],
                               cmap=plt.get_cmap('jet'),
                               vmin=sn_min,
                               vmax=sn_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[3][6])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # sum to gauss
#        im = axes[3][7].imshow(oiii_grid[6],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=g_min,
#                               vmax=g_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[3][7])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # velocity error
#        im = axes[3][8].imshow(oiii_grid[7],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=er_min,
#                               vmax=er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[3][8])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # sigma error
#        im = axes[3][9].imshow(oiii_grid[8],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=sig_er_min,
#                               vmax=sig_er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[3][9])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)


        # now plotting the apperture spectrum in 8th position instead

        axes[3][7].plot(oiii_int_wave,
                        oiii_int_spec,
                        drawstyle='steps-mid',
                        lw=2)

        axes[3][7].axhline(y=0E-19,
                           xmin=0,
                           xmax=1,
                           color='black',
                           ls='--',
                           lw=2)
        for ranges in h_sky_dict.values():
            axes[3][7].axvspan(ranges[0],
                               ranges[1],
                               alpha=0.5,
                               color='grey')
        axes[3][7].set_xlim(oiii_int_wave[0],
                            oiii_int_wave[-1])


        try:
            vel_min, vel_max = np.nanpercentile(ha_grid[1],[15.0, 85.0])
        except TypeError:
            vel_min, vel_max = [-100, 100]
        try:
            sig_min, sig_max = np.nanpercentile(ha_grid[2],[15.0, 85.0])
        except TypeError:
            sig_min, sig_max = [0, 150]
        try:
            flux_min, flux_max = np.nanpercentile(ha_grid[0],[15.0, 85.0])
        except TypeError:
            flux_min, flux_max = [1E-20, 5E-18]
        try:
            s_min, s_max = np.nanpercentile(ha_grid[3],[15.0, 85.0])
        except TypeError:
            s_min, s_max = [1E-20, 5E-18]
        try:
            noise_min, noise_max = np.nanpercentile(ha_grid[4],[15.0, 85.0])
        except TypeError:
            noise_min, noise_max = [1E-20, 5E-18]
        try:
            sn_min, sn_max = np.nanpercentile(ha_grid[5],[15.0, 85.0])
        except TypeError:
            sn_min, sn_max = [0, 10]
        try:
            g_min, g_max = np.nanpercentile(ha_grid[6],[15.0, 85.0])
        except TypeError:
            g_min, g_max = [0, 1.5]
        try:
            er_min, er_max = np.nanpercentile(ha_grid[7],[15.0, 85.0])
        except TypeError:
            er_min, er_max = [0, 100]
        try:
            sig_er_min, sig_er_max = np.nanpercentile(ha_grid[8],[15.0, 85.0])
        except TypeError:
            sig_er_min, sig_er_max = [0, 100]

        # set the title
        axes[4][0].set_title('HST F125W',
                             fontsize=15,
                             fontweight='bold')

        #  extraction stamp
        axes[4][0].imshow(extraction_stamp_5,
                          interpolation='nearest',
                          cmap=plt.cm.nipy_spectral)

            # flux
        im = axes[4][1].imshow(ha_grid[0],
                               cmap=plt.get_cmap('jet'),
                               vmin=flux_min,
                               vmax=flux_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[4][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # velocity
        im = axes[4][2].imshow(ha_grid[1],
                               cmap=plt.get_cmap('jet'),
                               vmin=vel_min,
                               vmax=vel_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[4][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # dispersion
        im = axes[4][3].imshow(ha_grid[2],
                               cmap=plt.get_cmap('jet'),
                               vmin=sig_min,
                               vmax=sig_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[4][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal
        im = axes[4][4].imshow(ha_grid[3],
                               cmap=plt.get_cmap('jet'),
                               vmin=s_min,
                               vmax=s_max,
                               interpolation='nearest')
        # plot also the extraction aperture
        axes[4][4].contour(masks_im[0],
                           masks_im[1],
                           masks_im[2],
                           1,
                           linewidths=3)

        # add colourbar to each plot
        divider = make_axes_locatable(axes[4][4])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # CONTINUUM
        im = axes[4][5].imshow(ha_cont,
                               cmap=plt.cm.nipy_spectral,
                               vmin=ha_cont_vmin,
                               vmax=ha_cont_vmax,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[4][5])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal to noise
        im = axes[4][6].imshow(ha_grid[5],
                               cmap=plt.get_cmap('jet'),
                               vmin=sn_min,
                               vmax=sn_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[4][6])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # sum to gauss
#        im = axes[4][7].imshow(ha_grid[6],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=g_min,
#                               vmax=g_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[4][7])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # velocity error
#        im = axes[4][8].imshow(ha_grid[7],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=er_min,
#                               vmax=er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[4][8])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # sigma error
#        im = axes[4][9].imshow(ha_grid[8],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=sig_er_min,
#                               vmax=sig_er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[4][9])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)


        # now plotting the apperture spectrum in 8th position instead

        axes[4][7].plot(ha_int_wave,
                        ha_int_spec,
                        drawstyle='steps-mid',
                        lw=2)
        axes[4][7].axhline(y=0E-19,
                           xmin=0,
                           xmax=1,
                           color='black',
                           ls='--',
                           lw=2)
        for ranges in k_sky_dict.values():
            axes[4][7].axvspan(ranges[0],
                               ranges[1],
                               alpha=0.5,
                               color='grey')
        axes[4][7].set_xlim(ha_int_wave[0],
                            ha_int_wave[-1])

        try:
            vel_min, vel_max = np.nanpercentile(nii_grid[1],[15.0, 85.0])
        except TypeError:
            vel_min, vel_max = [-100, 100]
        try:
            sig_min, sig_max = np.nanpercentile(nii_grid[2],[15.0, 85.0])
        except TypeError:
            sig_min, sig_max = [0, 150]
        try:
            flux_min, flux_max = np.nanpercentile(nii_grid[0],[15.0, 85.0])
        except TypeError:
            flux_min, flux_max = [1E-20, 5E-18]
        try:
            s_min, s_max = np.nanpercentile(nii_grid[3],[15.0, 85.0])
        except TypeError:
            s_min, s_max = [1E-20, 5E-18]
        try:
            noise_min, noise_max = np.nanpercentile(nii_grid[4],[15.0, 85.0])
        except TypeError:
            noise_min, noise_max = [1E-20, 5E-18]
        try:
            sn_min, sn_max = np.nanpercentile(nii_grid[5],[15.0, 85.0])
        except TypeError:
            sn_min, sn_max = [0, 10]
        try:
            g_min, g_max = np.nanpercentile(nii_grid[6],[15.0, 85.0])
        except TypeError:
            g_min, g_max = [0, 1.5]
        try:
            er_min, er_max = np.nanpercentile(nii_grid[7],[15.0, 85.0])
        except TypeError:
            er_min, er_max = [0, 100]
        try:
            sig_er_min, sig_er_max = np.nanpercentile(nii_grid[8],[15.0, 85.0])
        except TypeError:
            sig_er_min, sig_er_max = [0, 100]

        # set the title
        axes[5][0].set_title('HST F160W',
                             fontsize=15,
                             fontweight='bold')

        #  extraction stamp
        axes[5][0].imshow(extraction_stamp_6,
                          interpolation='nearest',
                          cmap=plt.cm.nipy_spectral)

        # flux
        im = axes[5][1].imshow(nii_grid[0],
                               cmap=plt.get_cmap('jet'),
                               vmin=flux_min,
                               vmax=flux_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[5][1])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # velocity
        im = axes[5][2].imshow(nii_grid[1],
                               cmap=plt.get_cmap('jet'),
                               vmin=vel_min,
                               vmax=vel_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[5][2])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # dispersion
        im = axes[5][3].imshow(nii_grid[2],
                               cmap=plt.get_cmap('jet'),
                               vmin=sig_min,
                               vmax=sig_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[5][3])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal
        im = axes[5][4].imshow(nii_grid[3],
                               cmap=plt.get_cmap('jet'),
                               vmin=s_min,
                               vmax=s_max,
                               interpolation='nearest')
        # plot also the extraction aperture
        axes[5][4].contour(masks_im[0],
                           masks_im[1],
                           masks_im[2],
                           1,
                           linewidths=3)

        # add colourbar to each plot
        divider = make_axes_locatable(axes[5][4])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # CONTINUUM
        im = axes[5][5].imshow(nii_cont,
                               cmap=plt.cm.nipy_spectral,
                               vmin=nii_cont_vmin,
                               vmax=nii_cont_vmax,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[5][5])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # signal to noise
        im = axes[5][6].imshow(nii_grid[5],
                               cmap=plt.get_cmap('jet'),
                               vmin=sn_min,
                               vmax=sn_max,
                               interpolation='nearest')

        # add colourbar to each plot
        divider = make_axes_locatable(axes[5][6])
        cax_new = divider.append_axes('right', size='10%', pad=0.05)
        plt.colorbar(im, cax=cax_new)

        # sum to gauss
#        im = axes[5][7].imshow(nii_grid[6],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=g_min,
#                               vmax=g_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[5][7])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # velocity error
#        im = axes[5][8].imshow(nii_grid[7],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=er_min,
#                               vmax=er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[5][8])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)
#        # sigma error
#        im = axes[5][9].imshow(nii_grid[8],
#                               cmap=plt.get_cmap('jet'),
#                               vmin=sig_er_min,
#                               vmax=sig_er_max,
#                               interpolation='nearest')
#        # add colourbar to each plot
#        divider = make_axes_locatable(axes[5][9])
#        cax_new = divider.append_axes('right', size='10%', pad=0.05)
#        plt.colorbar(im, cax=cax_new)


        # now plotting the apperture spectrum in 8th position instead

        axes[5][7].plot(nii_int_wave,
                        nii_int_spec,
                        drawstyle='steps-mid',
                        lw=2)

        axes[5][7].axhline(y=0E-19,
                           xmin=0,
                           xmax=1,
                           color='black',
                           ls='--',
                           lw=2)
        for ranges in k_sky_dict.values():
            axes[5][7].axvspan(ranges[0],
                               ranges[1],
                               alpha=0.5,
                               color='grey')
        axes[5][7].set_xlim(nii_int_wave[0],
                            nii_int_wave[-1])

        fig.tight_layout()
        # plt.show()
        save_name = '/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/ALL_LINE_GRIDS/blurred_0.2_without_3_sig_mask/' + gal_name + '.pdf'
        fig.savefig(save_name)
        plt.close('all')

multi_vel_field_all_lines('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Kband/KLP_K_NAMES_FINAL.txt',
                          '/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Hband/KLP_H_NAMES_FINAL.txt',
                          '/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES_FINAL.txt',
                          3.0,
                          0.25,
                          1.75,
                          ntimes=200,
                          spatial_smooth=True,
                          spectral_smooth=False,
                          smoothing_psf=0.2,
                          spectral_smooth_width=2,
                          prog='klp')

