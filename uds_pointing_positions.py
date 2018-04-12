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

position_file_name = '/disk2/turner/disk2/turner/DATA/KLP/CATALOGUES/NEW_POINTINGS/KARMA_WORK/UDS1/uds1_karma.cat'

with open(position_file_name, 'a') as f:

    reconstructed_list = glob.glob('SCI_RECONSTRUCTED*_Corrected.fits')
    first_file = reconstructed_list[0]

    first_table = fits.open(first_file)

    for i in range(1,25):
        first_header = first_table[i].header
        first_ra = first_header['CRVAL1']
        first_dec = first_header['CRVAL2']
        key = 'HIERARCH ESO OCS ARM' + str(i) + ' NAME'
        try:
            first_name = first_header[key]
        except KeyError:
            first_name = 'EMPTY_IFU'

        f.write('%s\t%s\t%s' % (first_name, first_ra, first_dec))
        f.write('\n')

f.close()