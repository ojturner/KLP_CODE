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

def make_table(sci_dir,
               glob_extension,
               names_file,
               star1,
               star2,
               star3):

    """
    Def:
    From the combined science directories and the KLP master file
    read in the object names, match with the name in the KLP master file
    to get the redshift, get the object IFU number so that it can be
    associated with the correct sky arm and use the number to associate
    with the standard star which was used to calibrate the object position
    """

    # read in the master file and get redshift and name
    k3d_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/K3D_ALL_OBJECTS.txt')
    k3d_names = k3d_table['Name']
    k3d_redshift = k3d_table['redshift']

    # what to search for in glob
    glob_search = sci_dir + '/COMBINE_SCI_RECONSTRUCTED*_' + glob_extension + '_*.fits'

    # loop over the combined objects in the science directory
    # and do the magic then write out each line to the names_file
    with open(names_file, 'a') as f:

        for entry in glob(glob_search):

            print entry
            # find just the galaxy name
            # and remove 'COMBINE_SCI_RECONSTRUCTED'
            gal_name = entry[len(entry) - entry[::-1].find("/"):]
            gal_name = gal_name[26:-5]

            # get the KLP master file index containing the
            # above string and hence the object redshift
            index = [i for i, s in enumerate(k3d_names) if gal_name in s][0]

            # now get the IFUNr for the sky frame and the correct
            # standard star
            cube = cubeOps(entry)
            ifu_nr = cube.IFUNR

            sky_file_name = sci_dir + '/COMBINE_SCI_RECONSTRUCTED_ARM' + str(ifu_nr) + '_SKY.fits'

            if ifu_nr > 0 and ifu_nr <= 8:
                std_name = sci_dir + '/COMBINE_SCI_RECONSTRUCTED_' + star1 + '.fits'
            elif ifu_nr > 8 and ifu_nr <= 16:
                std_name = sci_dir + '/COMBINE_SCI_RECONSTRUCTED_' + star2 + '.fits'
            else:
                std_name = sci_dir + '/COMBINE_SCI_RECONSTRUCTED_' + star3 + '.fits'

            f.write('%s\t%s\t%s\t%s\n' % (entry,
                                          k3d_redshift[index],
                                          std_name,
                                          sky_file_name))
    f.close()


make_table('/disk2/turner/disk2/turner/DATA/KLP/YJ/GP2/goods_p2_comb_calibrated',
           'GS4',
           '/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES.txt',
           'GS3str_31679',
           'C_STARS_16422',
           'C_STARS_20128')