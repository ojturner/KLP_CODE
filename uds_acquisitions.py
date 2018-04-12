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

position_file_name = '/disk2/turner/disk2/turner/DATA/KLP/CATALOGUES/NEW_POINTINGS/KARMA_WORK/UDS1/uds1_acquisitions.cat'

with open(position_file_name, 'a') as f:

    first_table = fits.open('/disk2/turner/disk2/turner/DATA/KLP/CATALOGUES/NEW_POINTINGS/KARMA_WORK/UDS1/KMOS.2013-12-09T04:01:26.064.fits')
    main_header = first_table[0].header

    for i in range(1,25):
        ra_key = 'HIERARCH ESO OCS ARM' + str(i) + ' ALPHA'
        dec_key = 'HIERARCH ESO OCS ARM' + str(i) + ' DELTA'
        name_key = 'HIERARCH ESO OCS ARM' + str(i) + ' NAME'
        
        try:
            ra = main_header[ra_key]
            dec = main_header[dec_key]
            name = main_header[name_key]
        except KeyError:
            ra = '99999999'
            dec = '9999999'
            name = 'EMPTY_IFU'

        f.write('%s\t%s\t%s' % (name, ra, dec))
        f.write('\n')

f.close()