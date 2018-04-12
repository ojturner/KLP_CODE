# This contains the functions necessary for fitting
# the integrated spectrum within object datacubes


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
import astropy.units as u
from spectral_cube import SpectralCube

# add the class file to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk2/turner/disk2/turner/DATA/KLP/CODE')

import flatfield_cube as f_f
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
import integrated_spectrum_fitting as spec_fit
import mask_sky as mask_the_sky
import klp_spaxel_fitting as klp_spax_fit

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', weight='bold')
rc('text', usetex=True)
rc('axes', linewidth=2)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# KBSS-MOSFIRE measurements
kbss_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/KBSS_MOSFIRE_LINE_RATIOS/steidel14_line_ratios.txt')
kbss_oiii_hb = kbss_table['log([OIII]/Hb)']
kbss_oiii_hb_lower_error = kbss_table['e_log([OIII]/Hb)']
kbss_oiii_hb_upper_error = kbss_table['E_log([OIII]/Hb)']
kbss_nii_ha = kbss_table['log([NII]/Ha)']
kbss_nii_ha_lower_error = kbss_table['e_log([NII]/Ha)']
kbss_nii_ha_upper_error = kbss_table['E_log([NII]/Ha)']


# sdss flux measurements
sdss_table = fits.open('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/SDSS_LINE_FLUXES/sdss_reliable_flux_measurements.fits')
sdss_oiii_hb = sdss_table[1].data['log_oiii_hb']
sdss_nii_ha = sdss_table[1].data['log_nii_ha']

# KLEVER Measurements
klever_table = ascii.read('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/DS9_REGIONS/klever_integrated_line_fluxes.txt')
klever_oiii_hb = klever_table['LOG_OIII/HB']
klever_oiii_hb_lower_error = klever_table['LOG_OIII/HB_LOWER_ERROR']
klever_oiii_hb_upper_error = klever_table['LOG_OIII/HB_UPPER_ERROR']
klever_nii_ha = klever_table['LOG_NII/HA']
klever_nii_ha_lower_error = klever_table['LOG_NII/HA_LOWER_ERROR']
klever_nii_ha_upper_error = klever_table['LOG_NII/HA_UPPER_ERROR']

fig, ax = plt.subplots(1,1, figsize=(10,10))

ax.tick_params(axis='both',
               which='major',
               labelsize=26,
               length=12,
               width=4)
ax.tick_params(axis='both', 
               which='minor',
               labelsize=26,
               length=6,
               width=4)

ax.set_xlabel(r'log([NII] $\lambda$6585/H$\alpha$)',
              fontsize=30,
              fontweight='bold',
              labelpad=15)

ax.set_ylabel(r'log([OIII] $\lambda$5008/H$\beta$)',
              fontsize=30,
              fontweight='bold',
              labelpad=15)

lines = {'linestyle': 'None'}
plt.rc('lines', **lines)

cm = plt.cm.get_cmap('Blues')

# try creating contour plot instead

xedges = np.arange(-1.6,0.6,0.01)
yedges = np.arange(-1.1,1.1,0.01)

H, xedges, yedges = np.histogram2d(sdss_nii_ha,sdss_oiii_hb,bins=[xedges,yedges])

diff_x = xedges[1] - xedges[0]
diff_y = yedges[1] - yedges[0]

xedges = xedges[:-1] + (diff_x / 2.0)
yedges = yedges[:-1] + (diff_y / 2.0)

X,Y = np.meshgrid(xedges,yedges)

CS_blue = ax.contour(X,Y,H.T,
                     levels=np.arange(5,150,5.0),
                     cmap=cm,
                     label='SDSS z < 0.2 reliable Brinchmann+06')

[i.set_linewidth(4.0) for i in ax.spines.itervalues()]

for entry in CS_blue.collections:
    plt.setp(entry,linewidth=2.75)

# plotting the KBSS redshift 2 points
p1_all = ax.errorbar(kbss_nii_ha,
            kbss_oiii_hb,
            ecolor='grey',
            xerr=[kbss_nii_ha_lower_error,kbss_nii_ha_upper_error],
            yerr=[kbss_oiii_hb_lower_error,kbss_oiii_hb_upper_error],
            marker='o',
            markersize=4,
            markerfacecolor='green',
            markeredgecolor='green',
            markeredgewidth=2,
            capsize=2,
            elinewidth=2,
            alpha=0.5,
            label='KBSS-MOSFIRE: Steidel+14')
klever_errorbar = ax.errorbar(klever_nii_ha,
                             klever_oiii_hb,
                            ecolor='black',
                            xerr=[klever_nii_ha_lower_error,klever_nii_ha_upper_error],
                            yerr=[klever_oiii_hb_lower_error,klever_oiii_hb_upper_error],
                            marker='*',
                            markersize=20,
                            markerfacecolor='red',
                            markeredgecolor='red',
                            markeredgewidth=2,
                            capsize=3,
                            elinewidth=3,
                            label='KLEVER')
ax.legend(loc='best',
            prop={'size':18,'weight':'bold'},
            frameon=False,
            markerscale=1.2,
            numpoints=1)
ax.set_xlim(-1.8,0.13)
ax.set_ylim(-0.9,1.2)
fig.tight_layout()
plt.show()
fig.savefig('/disk2/turner/disk2/turner/DATA/KLP/ANALYSIS/DS9_REGIONS/KLEVER_INT_BPT.png')
plt.close('all')



