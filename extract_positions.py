import numpy as np
import matplotlib.pyplot as plt
import glob
from astropy.io import ascii
from astropy.io import fits
from scipy.stats import binned_statistic
from matplotlib import rc
import matplotlib.ticker as ticker
from lmfit import Model, Parameters
from matplotlib.ticker import ScalarFormatter
from pylab import MaxNLocator
from pylab import *
from matplotlib import pyplot


# simply open up the cubes and take the RA/DEC and name and print to file

position_file_name = '/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/UDS/all_object_positions.txt'

with open(position_file_name, 'a') as f:

    for cube in glob.glob('COMBINE_SCI_RECONSTRUCTED*'):
        header = fits.open(cube)[1].header
        ra = header['CRVAL1']
        dec = header['CRVAL2']
        name = header['EXTNAME'][:-5]
        f.write('%s\t%s\t%s' % (name, ra, dec))
        f.write('\n')

f.close()