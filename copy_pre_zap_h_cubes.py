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

def copy_pre_zaps(orig_dir,
                  copy_dir):

    globbing_name = orig_dir + '/COMBINE_SCI_RECONSTRUCTED_*.fits'

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

    for entry in clean_obj_list:

        print entry

        os.system('cp %s %s' % (entry,copy_dir))

copy_pre_zaps('/disk2/turner/disk2/turner/DATA/KLP/H/GP4/goods_p4_comb_calibrated',
              '/disk2/turner/disk2/turner/DATA/KLP/H/PRE_ZAP_H_CUBES_/GP4')
