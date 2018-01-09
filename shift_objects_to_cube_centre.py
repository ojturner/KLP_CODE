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
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import MaxNLocator
from numpy import poly1d
from sys import stdout
from matplotlib import rc
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
import klp_flatfield_to_find_centre as klp_ff
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

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

# look at just the Kband to check the performance
# once the mask limits are supplied

table_k = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES_WITH_CONTINUUM.txt')
names_k = table_k['Filename']
redshift_k = table_k['redshift']
cont_x_lower_k = table_k['cont_x_lower']
cont_x_upper_k = table_k['cont_x_upper']
cont_y_lower_k = table_k['cont_y_lower']
cont_y_upper_k = table_k['cont_y_upper']

# lists for the central x and y positions
central_x_position = []
central_y_position = []

for name, red, xl,xu,yl,yu in zip(names_k,
                                  redshift_k,
                                  cont_x_lower_k,
                                  cont_x_upper_k,
                                  cont_y_lower_k,
                                  cont_y_upper_k):
    
    cont_dict = klp_ff.flatfield(name,
                                 red,
                                 'ha',
                                 xl,
                                 xu,
                                 yl,
                                 yu)

    # assign the flatfielded continuum and
    # only fit the region defined in the mask
    flatfield_continuum_k = cont_dict['cont1']

    # fill in the nan values momentarily and
    # blur by PSF
    for i in range(flatfield_continuum_k.shape[0]):
        for j in range(flatfield_continuum_k.shape[1]):
            if np.isnan(flatfield_continuum_k[i,j]):
                flatfield_continuum_k[i,j] = np.random.normal(loc=0E-19,scale=1E-18) 

    flatfield_continuum_k = psf.blur_by_psf(flatfield_continuum_k,
                                            0.25,
                                            0.1,
                                            50)

    flatfield_continuum_k[0:xl,:] = np.nan
    flatfield_continuum_k[xu:,:] = np.nan
    flatfield_continuum_k[:,0:yl] = np.nan
    flatfield_continuum_k[:,yu:] = np.nan

    minimum = np.nanmin(flatfield_continuum_k)

    flatfield_continuum_k = flatfield_continuum_k + np.abs(minimum)

    # fit 2d gaussian to the resultant thing and plot
    fit_cont, fit_params = g2d.fit_gaussian(flatfield_continuum_k)
    print fit_params

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.imshow(flatfield_continuum_k)
    ax.contour(fit_cont)
    plt.show()
    plt.close('all')

    # now we have successfully found the continuum centre
    # for each of the 26 cubes in the YJ, H and K bands. 
    # it is important to do all of the shifting at this
    # stage so that we can be sure all further analysis is
    # done with the objects having a common centre

    # going to use esorex to do the subpixel shifting to the
    # real centre of the cube, which is given by the size of
    # the spatial dimension minus 1 divided by 2 because of
    # zero indexing in python. Also notice that the horizontal
    # and vertical shifting have opposite signs because of the
    # esorex and python conventions differing

    # continuum centre horizontal and vertical
    continuum_centre_hor = fit_params[2]
    continuum_centre_ver = fit_params[3]

    # cube centre horizontal and vertical
    cube_centre_hor = np.around(((flatfield_continuum_k.shape[1] - 1) / 2.0),decimals=3)
    cube_centre_ver = np.around(((flatfield_continuum_k.shape[0] - 1) / 2.0),decimals=3)
    
    # append these central positions to the cube centre lists
    central_x_position.append(cube_centre_ver)
    central_y_position.append(cube_centre_hor)

    # the size of the horizontal and vertical shifts can now
    # be computed
    shift_hor = (continuum_centre_hor - cube_centre_hor)/5.0
    shift_ver = (cube_centre_ver - continuum_centre_ver)/5.0

    print shift_hor, shift_ver
    # create the shifts string to be substituted into the 
    # esorex call

    shift_string = '"' + str(shift_hor) + ',' + str(shift_ver) + '"'

    print shift_string

    # now have everything required to work the esorex shift routine
    # Now execute the recipe
    os.system('esorex kmo_shift --ifu=0 --shifts=%s --extrapolate=FALSE %s' % (shift_string,name))

    # the file SHIFT.fits has now been created containing the centred object
    # first remove the existing file if it exists, then move the created
    # and centred object into position

    if os.path.isfile(name):

        os.system('rm %s' % name)

    os.system('mv SHIFT.fits %s' % name)

    print central_x_position,central_y_position

# everything should now be shifted and stored in the original locations
# not sure why the filesize of the shifted object is half that of
# the original but going to roll with it

# append the central x and y positions to the filenames
central_x_position = np.array(central_x_position)
central_y_position = np.array(central_y_position)

table_k['Central_x'] = central_x_position
table_k['Central_y'] = central_y_position
table_k.write('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_BOGUS.txt', format='ascii')