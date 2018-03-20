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
import glob


# create a little function which starts in a 
# given directory, applies the master telluric correction
# to all the noise cubes and datacubes with the given name
# in that directory

def master_h_telluric_correct(corr_dir):

    globbing_name = corr_dir + '/COMBINE_SCI_RECONSTRUCTED_*.fits'

    glob_list = glob.glob(globbing_name)

    clean_obj_list = copy(glob_list)

    clean_obj_list = [x for x in clean_obj_list if "_oiii_" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_oii_" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_ha_" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_hb_" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_nii_" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_oiii_weak_" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_oiiiweak_" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_error_spectrum.fits" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_sky.fits" not in x]
    clean_obj_list = [x for x in clean_obj_list if "_SKY.fits" not in x]
    clean_obj_list = [x for x in clean_obj_list if ".png" not in x]

    master_telluric_table = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/MASTER_TELLURIC/MASTER_TELLURIC_HHH.fits')

    telluric_spectrum = master_telluric_table[0].data

    for entry in clean_obj_list:

        table = fits.open(entry,mode='update')

        print entry

        for i in range(table[1].data.shape[1]):
            for j in range(table[1].data.shape[2]):
                table[1].data[0:1730,i,j] = table[1].data[0:1730,i,j]/telluric_spectrum[0:1730]

        for i in range(table[2].data.shape[1]):
            for j in range(table[2].data.shape[2]):
                table[2].data[0:1730,i,j] = table[2].data[0:1730,i,j]/telluric_spectrum[0:1730]

        table.flush()
        table.close()

master_h_telluric_correct('/disk2/turner/disk2/turner/DATA/KLP/H/MULTIPLE_OB_OBJECTS/GS4_40218')