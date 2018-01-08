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

table_k = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES.txt')
names_k = table_k['Filename']

# lists for the central x and y positions
x_upper_array = []
x_lower_array = []
y_upper_array = []
y_lower_array = []

for name in names_k:
    cube = cubeOps(name)
    x_upper = cube.data.shape[1] - 5
    x_upper_array.append(x_upper)
    x_lower_array.append(5)
    y_upper = cube.data.shape[2] - 5
    y_upper_array.append(y_upper)
    y_lower_array.append(5)

# append the central x and y positions to the filenames
x_upper_array = np.array(x_upper_array)
x_lower_array = np.array(x_lower_array)
y_upper_array = np.array(y_upper_array)
y_lower_array = np.array(y_lower_array)

table_k['mask_x_lower'] = x_lower_array
table_k['mask_x_upper'] = x_upper_array
table_k['mask_y_lower'] = y_lower_array
table_k['mask_y_upper'] = y_upper_array
table_k.write('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES.txt', format='ascii')