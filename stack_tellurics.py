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
from glob import glob

# add the class file to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

from cubeClass import cubeOps

save_name = '/disk2/turner/disk2/turner/DATA/KLP/H/MASTER_TELLURIC/MASTER_TELLURIC_HHH.fits'
telluric_directory = '/disk2/turner/disk2/turner/DATA/KLP/H/MASTER_TELLURIC/INDIVIDUAL_TELLURICS'

glob_path = telluric_directory + '/*.fits'

spec_list = []

for telluric_file in glob(glob_path):
    table = fits.open(telluric_file)
    for entry in table:
        if entry.data is not(None):
            spec_list.append(entry.data)

spec_list = np.array(spec_list)
spec_stack = np.nanmedian(spec_list,axis=0)

tell_hdu = fits.PrimaryHDU(spec_stack)
tell_hdu.writeto(save_name,clobber=True)