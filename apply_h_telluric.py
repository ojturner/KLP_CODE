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


# create a little function which starts in a 
# given directory, applies the master telluric correction
# to all the noise cubes and datacubes with the given name
# in that directory

def master_h_telluric_correct(corr_dir):

    globbing_name = corr_dir + '/COMBINE_SCI_RECONSTRUCTED_*.fits'

    master_telluric_table = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/MASTER_TELLURIC/MASTER_TELLURIC_HHH.fits')

    telluric_spectrum = master_telluric_table[0].data

    for entry in glob.glob(globbing_name):

        table = fits.open(entry,mode='update')

        print entry

        for i in range(table[1].data.shape[1]):
            for j in range(table[1].data.shape[2]):
                table[1].data[:,i,j] = table[1].data[:,i,j]/telluric_spectrum

        for i in range(table[2].data.shape[1]):
            for j in range(table[2].data.shape[2]):
                table[2].data[:,i,j] = table[2].data[:,i,j]/telluric_spectrum

        table.flush()
        table.close()

master_h_telluric_correct('/disk2/turner/disk2/turner/DATA/KLP/H/GP4/goods_p4_comb_calibrated')