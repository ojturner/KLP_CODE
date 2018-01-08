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

# establishing the largest spatial x and y dimensions from the available
# wavebands (need to CHECK which wavebands exist from the parent K table)
# and then adding to the cubes in all three wavebands accordingly so that
# they end up with the same spatial dimensions. Next step is to measure
# the continuum centre in each waveband and move to the centre of this
# homogenised cube

kband_names = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Kband/KLP_K_NAMES.txt')['Filename']
hband_names = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/Hband/KLP_H_NAMES.txt')['Filename']
yjband_names = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/YJband/KLP_YJ_NAMES.txt')['Filename']

# start with each entry in the kband list of files
# and check whether it exists in the hband and yjband
# which are in principle the same list so it definitely should!

double_list = []

for entry in kband_names:
    # find just the galaxy name
    # and remove 'COMBINE_SCI_RECONSTRUCTED'
    gal_name = entry[len(entry) - entry[::-1].find("/"):]
    gal_name = gal_name[26:-5]

    # does it exist in the Hband
    index = [i for i, s in enumerate(hband_names) if gal_name in s]

    if not index:
        print 'OBJECT NOT FOUND IN KLP HBAND'
    elif len(index) == 1:
        print 'SINGLE OBSERVATION'
        # now need to determine the largest spatial dimension
        # and add that many nan rows/columns to the other cubes
        print entry
        print hband_names[index[0]]
        print yjband_names[index[0]]
        # first deal with the vertical direction
        table_k = fits.open(entry,mode='update')
        data_k = table_k[1].data
        data_k_noise = table_k[2].data
        table_h = fits.open(hband_names[index[0]],mode='update')
        data_h = table_h[1].data
        data_h_noise = table_h[2].data
        table_yj = fits.open(yjband_names[index[0]],mode='update')
        data_yj = table_yj[1].data
        data_yj_noise = table_yj[2].data
        # find the maximum vertical dimension
        max_vertical = np.nanmax((data_k.shape[1],data_h.shape[1],data_yj.shape[1]))
        print max_vertical
        # deal with each band in turn
        # it's not important to know which has the max size
        # first the k_band
        while data_k.shape[1] < max_vertical:
            new_data_k = []
            for face in data_k:
                new_data_k.append(np.vstack((face,np.repeat(np.nan,data_k.shape[2]))))
            data_k = np.array(new_data_k)
        while data_k_noise.shape[1] < max_vertical:
            new_data_k_noise = []
            for face in data_k_noise:
                new_data_k_noise.append(np.vstack((face,np.repeat(np.nan,data_k_noise.shape[2]))))
            data_k_noise = np.array(new_data_k_noise)

        # Next the h_band
        while data_h.shape[1] < max_vertical:
            new_data_h = []
            for face in data_h:
                new_data_h.append(np.vstack((face,np.repeat(np.nan,data_h.shape[2]))))
            data_h = np.array(new_data_h)
        while data_h_noise.shape[1] < max_vertical:
            new_data_h_noise = []
            for face in data_h_noise:
                new_data_h_noise.append(np.vstack((face,np.repeat(np.nan,data_h_noise.shape[2]))))
            data_h_noise = np.array(new_data_h_noise)

        # Finally the YJ band
        while data_yj.shape[1] < max_vertical:
            new_data_yj = []
            for face in data_yj:
                new_data_yj.append(np.vstack((face,np.repeat(np.nan,data_yj.shape[2]))))
            data_yj = np.array(new_data_yj)
        while data_yj_noise.shape[1] < max_vertical:
            new_data_yj_noise = []
            for face in data_yj_noise:
                new_data_yj_noise.append(np.vstack((face,np.repeat(np.nan,data_yj_noise.shape[2]))))
            data_yj_noise = np.array(new_data_yj_noise)

        # now play the exact same game with the horizontal direction.
        # programmatically easier because we can make use of dstack
        max_horizontal = np.nanmax((data_k.shape[2],data_h.shape[2],data_yj.shape[2]))
        print max_horizontal
        base_vert_vec = np.repeat(np.nan,max_vertical)
        vert_vec = []
        for i in range(2048):
            vert_vec.append(base_vert_vec)
        vert_vec = np.array(vert_vec)

        # now everything should have first axis equal to max vertical
        # construct an array on nans to dstack with each array
        # first the Kband
        while data_k.shape[2] < max_horizontal:
            data_k = np.dstack((data_k,vert_vec))
        while data_k_noise.shape[2] < max_horizontal:
            data_k_noise = np.dstack((data_k_noise,vert_vec))
        # Hband
        while data_h.shape[2] < max_horizontal:
            data_h = np.dstack((data_h,vert_vec))
        while data_h_noise.shape[2] < max_horizontal:
            data_h_noise = np.dstack((data_h_noise,vert_vec))
        # YJband
        while data_yj.shape[2] < max_horizontal:
            data_yj = np.dstack((data_yj,vert_vec))
        while data_yj_noise.shape[2] < max_horizontal:
            data_yj_noise = np.dstack((data_yj_noise,vert_vec))

        # everything should be homogeneous and equal to the
        # max dimensions of the cube across the three wavebands
        # finally flush and close the manipulated datacubes
        table_k[1].data = data_k
        table_k[2].data = data_k_noise
        table_k.flush()
        table_k.close()
        table_h[1].data = data_h
        table_h[2].data = data_h_noise
        table_h.flush()
        table_h.close()
        table_yj[1].data = data_yj
        table_yj[2].data = data_yj_noise
        table_yj.flush()
        table_yj.close()

    elif len(index) == 2:
        print 'DOUBLE OBSERVATION'
        double_list.append(gal_name)
    else:
        print 'MORE THAN DOUBLE??'

print double_list