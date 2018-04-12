import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.io import fits
from scipy.stats import binned_statistic
from matplotlib import rc
import matplotlib.ticker as ticker
from lmfit import Model, Parameters
from matplotlib.ticker import ScalarFormatter
from pylab import MaxNLocator
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot
import astropy.wcs.utils as autils
from astropy.wcs import WCS
from PIL import Image

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font', weight='bold')
rc('text', usetex=True)
rc('axes', linewidth=2)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

# create a figure with matplotlib showing a 7 x 1 grid of the narrowband
# images for GS3_26790
image_file = fits.open('/disk2/turner/disk1/turner/DATA/IMAGING/CANDELS_GOODSS/F160W/hlsp_hlf_hst_wfc3-60mas_goodss_f160w_v1.5_sci.fits')
# F160W
image_header = image_file[0].header
image_data = image_file[0].data
w = WCS(image_header)
lon, lat = w.wcs_world2pix(53.1745083333, -27.72536, 1)
lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
lon_unit = int(np.round((2.2 / (lon_scale * 3600)) / 2.0))
lat_unit = int(np.round((2.2 / (lat_scale * 3600)) / 2.0))
# and get the extraction stamps for sci and wht, ready to save
hst_extraction_stamp = image_data[lat - lat_unit:lat + lat_unit,
                                  lon - lon_unit:lon + lon_unit]

oii_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/YJ/GP1/goods_p1_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS3_26790_oii_signal_field.fits')[0].data[5:-5,5:-5]
hb_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/GP1/goods_p1_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS3_26790_hb_signal_field.fits')[0].data[5:-5,5:-5]
oiiiweak_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/GP1/goods_p1_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS3_26790_oiiiweak_signal_field.fits')[0].data[5:-5,5:-5]
oiii_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/GP1/goods_p1_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS3_26790_oiii_signal_field.fits')[0].data[5:-5,5:-5]
ha_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/GS/K/P1/p1_comb/COMBINE_SCI_RECONSTRUCTED_GS3_26790_ha_signal_field.fits')[0].data[5:-5,5:-5]
nii_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/GS/K/P1/p1_comb/COMBINE_SCI_RECONSTRUCTED_GS3_26790_nii_signal_field.fits')[0].data[5:-5,5:-5]

fig, axes = plt.subplots(1, 7, figsize=(20, 4))

# set the limits
oii_min = 0E-19
oii_max = 3E-19

# box thickness
for j in range(7):
    [i.set_linewidth(3.0) for i in axes[j].spines.itervalues()]

# get rid of pesky tick labels
for entry in axes:
    entry.set_xticks([])
    entry.set_yticks([])

plot_fontsize = 20
# set the title
axes[0].set_title('HST F160W',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[1].set_title('[OII]3727',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[2].set_title(r'H$\beta$',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[3].set_title('[OIII]4960',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[4].set_title('[OIII]5007',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[5].set_title(r'H$\alpha$',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[6].set_title('[NII]6584',
                  fontsize=plot_fontsize,
                  fontweight='bold')
# extraction stamp
axes[0].imshow(hst_extraction_stamp,
               interpolation='nearest',
               cmap=plt.cm.nipy_spectral)


# signals
im = axes[1].imshow(oii_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[2].imshow(hb_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[3].imshow(oiiiweak_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[4].imshow(oiii_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[5].imshow(ha_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[6].imshow(nii_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

fig.tight_layout()
plt.show()
fig.savefig('/disk2/turner/disk2/turner/DATA/KLP/CATALOGUES/FOLLOW_UP/MUSE_OVERLAP/GS3_26790.png')
plt.close('all')

# F160W
image_header = image_file[0].header
image_data = image_file[0].data
w = WCS(image_header)
lon, lat = w.wcs_world2pix(53.11245,-27.69263, 1)
lon_scale, lat_scale = autils.proj_plane_pixel_scales(w)
lon_unit = int(np.round((2.2 / (lon_scale * 3600)) / 2.0))
lat_unit = int(np.round((2.2 / (lat_scale * 3600)) / 2.0))
# and get the extraction stamps for sci and wht, ready to save
hst_extraction_stamp = image_data[lat - lat_unit:lat + lat_unit,
                                  lon - lon_unit:lon + lon_unit]

oii_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/YJ/GP2/goods_p2_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS4_46432_oii_signal_field.fits')[0].data[5:-5,5:-5]
hb_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/GP2/goods_p2_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS4_46432_hb_signal_field.fits')[0].data[5:-5,5:-5]
oiiiweak_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/GP2/goods_p2_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS4_46432_oiiiweak_signal_field.fits')[0].data[5:-5,5:-5]
oiii_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/H/GP2/goods_p2_comb_calibrated/COMBINE_SCI_RECONSTRUCTED_GS4_46432_oiii_signal_field.fits')[0].data[5:-5,5:-5]
ha_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/GS/K/P6/p6_comb/COMBINE_SCI_RECONSTRUCTED_GS4_46432_ha_signal_field.fits')[0].data[5:-5,5:-5]
nii_nband = fits.open('/disk2/turner/disk2/turner/DATA/KLP/KMOS_3D_DATA/GS/K/P6/p6_comb/COMBINE_SCI_RECONSTRUCTED_GS4_46432_nii_signal_field.fits')[0].data[5:-5,5:-5]

fig, axes = plt.subplots(1, 7, figsize=(20, 4))

# set the limits
oii_min = 0E-19
oii_max = 5E-19

# box thickness
for j in range(7):
    [i.set_linewidth(3.0) for i in axes[j].spines.itervalues()]

# get rid of pesky tick labels
for entry in axes:
    entry.set_xticks([])
    entry.set_yticks([])

plot_fontsize = 20
# set the title
axes[0].set_title('HST F160W',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[1].set_title('[OII]3727',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[2].set_title(r'H$\beta$',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[3].set_title('[OIII]4960',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[4].set_title('[OIII]5007',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[5].set_title(r'H$\alpha$',
                  fontsize=plot_fontsize,
                  fontweight='bold')
axes[6].set_title('[NII]6584',
                  fontsize=plot_fontsize,
                  fontweight='bold')
# extraction stamp
axes[0].imshow(hst_extraction_stamp,
               interpolation='nearest',
               cmap=plt.cm.nipy_spectral)


# signals
im = axes[1].imshow(oii_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[2].imshow(hb_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[3].imshow(oiiiweak_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[4].imshow(oiii_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[5].imshow(ha_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

# signals
im = axes[6].imshow(nii_nband,
                       cmap=plt.get_cmap('jet'),
                       vmin=oii_min,
                       vmax=oii_max,
                       interpolation='nearest')

fig.tight_layout()
plt.show()
fig.savefig('/disk2/turner/disk2/turner/DATA/KLP/CATALOGUES/FOLLOW_UP/MUSE_OVERLAP/GS4_46432.png')
plt.close('all')