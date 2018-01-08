from astropy.io import fits
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import copy


def flatfield(cube,
              redshift,
              line,
              xlower,
              xupper,
              ylower,
              yupper):
    
    """
    Def:
    Trying to extract continuum from the KMOS galaxies.
    First plot the continuum without flatfielding, then 
    subtract the median flux from every row and column of
    each slice of the masked cube to flatfield correctly
    and plot the result. 

    Input:
            cube - The cube object to perform operation on
            m_data - data for constructing the mask

    Output:
            cube_flatfielded - new copy of the cube with the
                                flatfielding applied

    Notes: Also want to exclude the edge spaxels from the analysis as these 
            contain different exposure times to the rest of the cube. 

    """
    # choose which line we're dealing with
    if line == 'oiii':
        line_wavelength = 0.500824
    elif line == 'ha':
        line_wavelength = 0.656461
    elif line == 'oii':
        line_wavelength = 0.37284

    #  line wavelength
    central_l = (1 + redshift) * line_wavelength

    # find the gal_name
    gal_name = cube[len(cube) -
                            cube[::-1].find("/"):][:-5]
    # read in the object 
    cube_table = fits.open(cube)

    print 'THIS IS THE CUBE NAME: %s' % cube

    # get the filter ID from the primary header
    prim_header = cube_table[0].header
    filt = prim_header['HIERARCH ESO INS FILT1 ID']

    data = cube_table[1].data

    data_header = cube_table[1].header

    start_l = data_header['CRVAL3']

    dl = data_header['CDELT3']

    wave_array = start_l + np.arange(0, 2048*dl, dl)

    o_peak = np.argmin(abs(central_l - wave_array))

    f_data = copy(data)

    # Now for each wavelength slice - flatfield using the mask limits
    # start with i, j > 3 in both cases to mitigate edge effects with lower
    # exposure times

    dim_x = f_data.shape[1]
    dim_y = f_data.shape[2]

    # create the mask and masked cube to take the medians from

    mask = np.ones(shape=(dim_x, dim_y))

    # every position near the edge and on top of the object is
    # masked with a np.nan value

    mask[0:4, :] = np.nan
    mask[-4:, :] = np.nan
    mask[:, 0:4] = np.nan
    mask[:, -4:] = np.nan

    mask[xlower:xupper,ylower:yupper] = np.nan

    # masked cube is given by multiplying the mask
    # with the cube and using the mapping function

    masked_data = f_data*mask[np.newaxis,:,:]

    # loop round each surface and subtract the median values
    for i in range(f_data.shape[0]):

        for x in range(3, f_data.shape[1] - 3):

            f_data[i, x, :] = f_data[i, x, :] - np.nanmedian(masked_data[i, x, :])

        for y in range(3, data.shape[2] - 3):

            f_data[i, :, y] = f_data[i, :, y] - np.nanmedian(masked_data[i, :, y])

    # conditional execution depending on waveband

    if filt == 'K':

        o_nband = np.nanmedian(f_data[o_peak-5:o_peak+5, :, :], axis=0)

        cont_1 = np.nanmedian(data[300:1600, :, :], axis=0)

        cont_2 = np.nanmedian(f_data[300:1600, :, :], axis=0)

    elif filt == 'HK':

        o_nband = np.nanmedian(f_data[o_peak-3:o_peak+3, :, :], axis=0)

        cont_1 = np.nanmedian(data[100:1800, :, :], axis=0)

        cont_2 = np.nanmedian(f_data[100:1800, :, :], axis=0)

    else:

        o_nband = np.nanmedian(f_data[o_peak-5:o_peak+5, :, :], axis=0)

        cont_1 = np.nanmedian(data[100:1800, :, :], axis=0)

        cont_2 = np.nanmedian(f_data[100:1800, :, :], axis=0)

    return {'cont1' : cont_1,
            'cont2' : cont_2,
            'LINE' : o_nband}

#c = '/Users/owenturner/DATA/uncalibrated_goods_p1_0.8_10_better/Science/combine_sci_reconstructed_bs008543.fits'
#v = fits.open('/Users/owenturner/DATA/uncalibrated_goods_p1_0.8_10_better/Science/combine_sci_reconstructed_bs008543_vel_field.fits')
#v = v[0].data
#flatfield(c, v, 3.473288)

