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
import glob

# add the class file to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

from cubeClass import cubeOps

def save_sky_spectra(comb_dir):

    # take each object in the comb_dir that is of type COMBINE
    # remove the sky arms, find the IFU number, find the sky corresponding
    # to that - open it up and median stack the spaxels and then 
    # save as a new fits file in the sky spectra folder which is one
    # folder back as a new fits file with the same name structure as 
    # the object

    new_name_list = []

    combine_names = comb_dir + '/COMBINE_SCI_RECONSTRUCTED_*.fits'

    for entry in glob.glob(combine_names):

        new_name_list.append(entry)

    # get down to just the objects instead of the sky files

    name_list = [x for x in new_name_list if 'COMBINE_SCI_RECONSTRUCTED_ARM' not in x]

    for entry in name_list:

        print entry

        cube = cubeOps(entry)

        ifu_nr = cube.IFUNR

        # find corresponding sky file 

        sky_file_name = comb_dir + '/COMBINE_SCI_RECONSTRUCTED_ARM' + str(ifu_nr) + '_SKY.fits'

        table = fits.open(sky_file_name)

        data = table[1].data

        one_d_sky_spec = np.nanmedian(np.nanmedian(data,axis=1),axis=1)

        # what is the save name

        if entry.find("/") == -1:

            gal_name = copy(entry)

        # Otherwise the directory structure is included and have to
        # search for the backslash and omit up to the last one

        else:

            gal_name = entry[len(entry) - entry[::-1].find("/"):]

        sky_spec_name = comb_dir + '/' + gal_name[:-5] + '_k_sky.fits'

        sky_hdu = fits.PrimaryHDU(one_d_sky_spec)

        sky_hdu.writeto(sky_spec_name,clobber=True)


save_sky_spectra('/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/UDS/K/P4/p4_comb')