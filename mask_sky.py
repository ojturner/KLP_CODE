# This contains the functions necessary for fitting
# the integrated spectrum within object datacubes


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

# add the class file to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/Class')

# add the functions folder to the PYTHONPATH
sys.path.append('/disk2/turner/disk1/turner/PhD'
                + '/KMOS/Analysis_Pipeline/Python_code/functions')

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

from cubeClass import cubeOps
from galPhysClass import galPhys
from vel_field_class import vel_field

# define speed of light
c = 2.99792458E5

def ret_k_sky():

    """
    Def:
    return a dictionary of values corresponding to the K-band sky
    """
    sky_dict = {1: [1.9508, 1.9532],
                2: [1.9552, 1.9568],
                3: [1.9588, 1.9601],
                4: [1.9613, 1.9625],
                5: [1.9635, 1.9650],
                6: [1.9672, 1.9685],
                7: [1.9694, 1.9710],
                8: [1.9731, 1.9743],
                9: [1.9745, 1.9759],
                10: [1.9764, 1.9781],
                11: [1.9834, 1.9847],
                12: [1.9888, 1.9908],
                13: [1.9921, 1.9930],
                14: [2.0000, 2.0200],
                15: [2.0269, 2.0283],
                16: [2.0316, 2.0323],
                17: [2.0333, 2.0347],
                18: [2.0405, 2.0422],
                19: [2.0493, 2.0506],
                20: [2.0556, 2.0572],
                21: [2.0723, 2.0736],
                22: [2.0855, 2.0866],
                23: [2.0904, 2.0915],
                24: [2.1010, 2.1123],
                25: [2.1151, 2.1161],
                26: [2.1172, 2.1183],
                27: [2.1228, 2.1238],
                28: [2.1245, 2.1255],
                29: [2.1274, 2.1284],
                30: [2.1313, 2.1331],
                31: [2.1500, 2.1516],
                32: [2.1532, 2.1549],
                33: [2.1575, 2.1585],
                34: [2.1633, 2.1642],
                35: [2.1707, 2.1718],
                36: [2.1759, 2.1765],
                37: [2.1797, 2.1808],
                38: [2.1868, 2.1878],
                39: [2.1950, 2.1962],
                40: [2.2047, 2.2058],
                41: [2.2120, 2.2133],
                42: [2.2243, 2.2252],
                43: [2.2307, 2.2319],
                44: [2.2335, 2.2350],
                45: [2.2456, 2.2465],
                46: [2.2513, 2.2524],
                47: [2.2684, 2.2649],
                48: [2.2737, 2.2747],
                49: [2.3162, 2.3187],
                50: [2.3396, 2.3423],
                51: [2.3700, 2.5000]}

    return sky_dict

def ret_yj_sky():

    """
    Def:
    return a dictionary of values corresponding to the K-band sky
    """
    sky_dict = {1: [1.02, 1.02165],
                2: [1.0285, 1.03],
                3: [1.03722, 1.0379],
                4: [1.04182, 1.04254],
                5: [1.04685, 1.04877],
                6: [1.07289, 1.07343],
                7: [1.0751, 1.07569],
                8: [1.08311, 1.08378],
                9: [1.08414, 1.0848],
                10:[1.08962, 1.09019],
                11:[1.09235, 1.09306],
                12:[1.0949, 1.09547],
                13:[1.09721, 1.09787],
                14:[1.1007, 1.10119],
                15:[1.10264, 1.10335],
                16:[1.10876, 1.10937],
                17:[1.12977, 1.13018],
                18:[1.13104, 1.13154],
                19:[1.13289, 1.13351],
                20:[1.13516, 1.13566],
                21:[1.1437, 1.14438],
                22:[1.14488, 1.14547],
                23:[1.14662, 1.14706],
                24:[1.15363, 1.15422],
                25:[1.1589, 1.1595],
                26:[1.16467, 1.16538],
                27:[1.17136, 1.17189],
                28:[1.20048,1.20107],
                29:[1.20224, 1.20341],
                30:[1.21192,1.2139],
                31:[1.21935, 1.22],
                32:[1.2226, 1.2233],
                33:[1.2255, 1.22611],
                34:[1.22842, 1.2291],
                35:[1.23237, 1.23292],
                36:[1.23486,1.23548],
                37:[1.2398, 1.24032],
                38:[1.2421, 1.24254],
                39:[1.26813, 1.28108],
                40:[1.2904, 1.29242],
                41:[1.30195, 1.30258],
                42:[1.30498, 1.30562],
                43:[1.30287, 1.30906],
                44:[1.31263, 1.31313],
                45:[1.31541, 1.31607]}

    return sky_dict

def ret_hk_sky():

    """
    Def:
    return a dictionary of values corresponding to the K-band sky
    """
    sky_dict = {1: [1.50458,1.50985],
                2: [1.51775,1.52006],
                3: [1.52302,1.52549],
                4: [1.52796,1.52993],
                5: [1.53207,1.53454],
                6: [1.53849,1.54047],
                7: [1.54228,1.54442],
                8: [1.54969,1.55199],
                9: [1.55331,1.56071],
                10: [1.56236,1.56648],
                11: [1.56911,1.57109],
                12: [1.58228,1.58804],
                13: [1.59643,1.59825],
                14: [1.60203,1.60434],
                15: [1.60713,1.60894],
                16: [1.61207,1.61421],
                17: [1.61882,1.62030],
                18: [1.62261,1.62475],
                19: [1.63051,1.65125],
                20: [1.65487,1.65602],
                21: [1.66063,1.66178],
                22: [1.66820,1.67413],
                23: [1.67512,1.67709],
                24: [1.68335,1.68516],
                25: [1.68944,1.69141],
                26: [1.69487,1.69635],
                27: [1.70014,1.70211],
                28: [1.70689,1.70870],
                29: [1.71166,1.71314],
                30: [1.71676,1.72187],
                31: [1.72417,1.72565],
                32: [1.72746,1.73964],
                33: [1.74228,1.74590],
                34: [1.74968,1.75133],
                35: [1.75232,1.75347],
                36: [1.76450,1.77042],
                37: [1.77273,1.77388],
                38: [1.78063,1.78195],
                39: [1.78738,1.78919],
                40: [1.79834,1.80045],
                41: [1.80623,1.80753],
                42: [1.81094,1.81281],
                43: [1.82054,1.82176],
                44: [1.82485,1.82623],
                45: [1.84534,1.84689],
                46: [1.85209,1.85339],
                47: [1.85843,1.85925],
                48: [1.87405,1.87502],
                49: [1.92406,1.92593],
                50: [1.93414,1.93577],
                51: [1.95140,1.95292],
                52: [1.95541,1.95665],
                53: [1.95900,1.95983],
                54: [1.96135,1.96508],
                55: [1.96729,1.97116],
                56: [1.97337,1.97821],
                57: [1.97945,1.98111],
                58: [1.98318,1.98484],
                59: [1.98649,1.99326],
                60: [1.99948,2.00459],
                61: [2.01882,2.02034],
                62: [2.02656,2.02877],
                63: [2.03305,2.03499],
                64: [2.04024,2.04231],
                65: [2.04936,2.05074],
                66: [2.05530,2.05765],
                67: [2.07188,2.07409],
                68: [2.08542,2.08680],
                69: [2.09025,2.09191],
                70: [2.10076,2.10418],
                71: [2.10564,2.10760],
                72: [2.10922,2.11231],
                73: [2.11524,2.11833],
                74: [2.12288,2.12581],
                75: [2.12727,2.12841],
                76: [2.13118,2.13329],
                77: [2.14972,2.15509],
                78: [2.15736,2.15883],
                79: [2.16322,2.16452],
                80: [2.17037,2.17167],
                81: [2.17574,2.17688],
                82: [2.17916,2.18111],
                83: [2.18647,2.18826],
                84: [2.19444,2.19672],
                85: [2.20453,2.20615],
                86: [2.21168,2.21364],
                87: [2.22437,2.22551],
                88: [2.23055,2.23234],
                89: [2.24551,2.24681],
                90: [2.25120,2.25267],
                91: [2.26815,2.26962],
                92: [2.27352,2.27491],
                93: [2.37000,2.50000]}

    return sky_dict

def ret_h_sky():

    """
    Def:
    return a dictionary of values corresponding to the K-band sky
    """
    sky_dict = {1: [1.45136, 1.45241],
                2: [1.45591, 1.45684],
                3: [1.45999, 1.46108],
                4: [1.46621, 1.46695],
                5: [1.46948, 1.47037],
                6: [1.47360, 1.47461],
                7: [1.47523, 1.47601],
                8: [1.47686, 1.47772],
                9: [1.47796, 1.47900],
                10:[1.47951, 1.48099],
                11:[1.48277, 1.48386],
                12:[1.48596, 1.48701],
                13:[1.48822, 1.48946],
                14:[1.49047, 1.49144],
                15:[1.49277, 1.49370],
                16:[1.50043, 1.50109],
                17:[1.50233, 1.50299],
                18:[1.50482, 1.50750],
                19:[1.50836, 1.50945],
                20:[1.51096, 1.51194],
                21:[1.51831, 1.51936],
                22:[1.52362, 1.52470],
                23:[1.52821, 1.52934],
                24:[1.53086, 1.53142],
                25:[1.53268, 1.53389],
                26:[1.53454, 1.53523],
                27:[1.53710, 1.54147],
                28:[1.54264, 1.54407],
                29:[1.54594, 1.54780],
                30:[1.54854, 1.54927],
                31:[1.54975, 1.55218],
                32:[1.55352, 1.55517],
                33:[1.55664, 1.55755],
                34:[1.55928, 1.56032],
                35:[1.56102, 1.56154],
                36:[1.56279, 1.56366],
                37:[1.56513, 1.56613],
                38:[1.56821, 1.56869],
                39:[1.56981, 1.57068],
                40:[1.57500, 1.59040],
                41:[1.59122, 1.59209],
                42:[1.59685, 1.59776],
                43:[1.60222, 1.60455],
                44:[1.60552, 1.60626],
                45:[1.60739, 1.60853],
                46:[1.61160, 1.61353],
                47:[1.61563, 1.61631],
                48:[1.61904, 1.61995],
                49:[1.62063, 1.62597],
                50:[1.62671, 1.62739],
                51:[1.62768, 1.62830],
                52:[1.62995, 1.63217],
                53:[1.63376, 1.63683],
                54:[1.63853, 1.64012],
                55:[1.64109, 1.64188],
                56:[1.64381, 1.64518],
                57:[1.64722, 1.64836],
                58:[1.64984, 1.65075],
                59:[1.65262, 1.65614],
                60:[1.65836, 1.65898],
                61:[1.66069, 1.66154],
                62:[1.66870, 1.66974],
                63:[1.67518, 1.67694],
                64:[1.68359, 1.68459],
                65:[1.68881, 1.68938],
                66:[1.68989, 1.69131],
                67:[1.69512, 1.69603],
                68:[1.70046, 1.70160],
                69:[1.70540, 1.70605],
                70:[1.70705, 1.70834],
                71:[1.70994, 1.71288],
                72:[1.71695, 1.71818],
                73:[1.71865, 1.71995],
                74:[1.72066, 1.72608],
                75:[1.72731, 1.72885],
                76:[1.73002, 1.73073],
                77:[1.73256, 1.73362],
                78:[1.73474, 1.73639],
                79:[1.73792, 1.73916],
                80:[1.74139, 1.74340],
                81:[1.74458, 1.74552],
                82:[1.74988, 1.75112],
                83:[1.75253, 1.75524],
                84:[1.76089, 1.76319],
                85:[1.76455, 1.76773],
                86:[1.76826, 1.77032],
                87:[1.77303, 1.77386],
                88:[1.77721, 1.77827],
                89:[1.78075, 1.78381],
                90:[1.78493, 1.78576],
                91:[1.78758, 1.78870],
                92:[1.79889, 1.86]}


    return sky_dict

def masking_sky(wave_array,
                spec,
                filt):

    """
    Def:
    Mask the sky lines so that the fit can be carried out without being
    contaminated by consideration of the skylines. The only problem is when
    the emission line should be right on top of a skyline. What would happen 
    if the centre of the line is contrained to lie in a region which has no
    data? 

    Input:
            wavelength - k-band wavelength array
            flux - corresponding flux values

    Output:
            wavelength_masked - np.masked_array version with sky masked
            flux_masked - as above, corresponding
    """

    # choose the appropriate sky dictionary for the filter
    if filt == 'YJ':
        #print 'YJ-band sky selected'
        sky_dict = ret_yj_sky()
    elif filt == 'H':
        #print 'H-band sky selected'
        sky_dict = ret_h_sky()
    elif filt == 'K':
        #print 'K-band sky selected'
        sky_dict = ret_k_sky()
    else:
        raise ValueError('Please ensure that you have'
                             + ' chosen an appropriate waveband')

    # now loop through and mask off all of the offending regions

    wave_array_masked = copy(wave_array)

    for entry in sky_dict.keys():

        wave_array_masked = ma.masked_where(
            np.logical_and(wave_array_masked > sky_dict[entry][0],
                           wave_array_masked < sky_dict[entry][1]),
            wave_array_masked, copy=True)

    # apply the final mask to the flux array

    spec_masked = ma.MaskedArray(spec,
                                 mask=wave_array_masked.mask)

#    fig, ax = plt.subplots(1,1,figsize=(18,8))
#    ax.plot(wave_array,spec_masked,drawstyle='steps-mid')
#    for ranges in sky_dict.values():
#        ax.axvspan(ranges[0],ranges[1],alpha=0.5,color='grey')
#    plt.show()
#    plt.close('all')

    return wave_array, spec_masked