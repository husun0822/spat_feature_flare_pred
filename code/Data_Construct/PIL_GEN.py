'''
This file is to replicate the PIL mask generation algorithm implemented in Wang et al. (2019). There are three
critical steps involved in the algorithm:

1. Positive/Negative bitmap generation (Threshold a Br image at +/- 200G)
2. DBSCAN clustering and locate the "core" points for two bitmaps
3. Convolve the image with Gaussian filter and generate the PIL mask

The original code is written by Jingjing Wang in IDL, and here we give a python implementation
'''

import h5py
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from sklearn.cluster import DBSCAN
import math
import numpy as np

def RoundToSigFigs_fp(x, sigfigs):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value has the same type as x.

    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value.
    """
    __logBase10of2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1
    if not ( type(sigfigs) is int or type(sigfigs) is np.long or
             isinstance(sigfigs, np.integer) ):
        raise TypeError( "RoundToSigFigs_fp: sigfigs must be an integer." )

    if sigfigs <= 0:
        raise ValueError( "RoundToSigFigs_fp: sigfigs must be positive." )

    if not np.isreal( x ):
        print(x)
        raise TypeError( "RoundToSigFigs_fp: x must be real." )

    xsgn = np.sign(x)
    absx = xsgn * x
    mantissa, binaryExponent = np.frexp( absx )

    decimalExponent = __logBase10of2 * binaryExponent
    omag = np.floor(decimalExponent)

    mantissa *= 10.0**(decimalExponent - omag)

    if mantissa < 1.0:
        mantissa *= 10.0
        omag -= 1.0

    return xsgn * np.around( mantissa, decimals=sigfigs - 1 ) * 10.0**omag


def Bitmap(image, threshold = 200):
    '''
    Generate Bitmap for positive and negative components.
    :param image: 2-d numpy array (Bz image)
    :param threshold: magnetic field threshold
    :return: a positive bitmap and a negative bitmap
    '''

    pos = (image >= threshold).astype(int)
    neg = (image <= -threshold).astype(int)

    return pos, neg


def Cluster(comp, eps = 30, portion = 0.125):
    '''
    Detect the "core" points for the component using DBSCAN, a density based clustering method.
    :param comp: 2-d numpy array, should be a bitmap
    :param eps: radius for checking the density of a point
    :param portion: number of neighbors needed within the radius for a point to be considered as "core" point
    :return: the core points map, same size as the input comp
    '''

    threshold = portion*(math.pi)*eps*eps
    kernel = np.zeros(shape = (2*eps+1, 2*eps+1))
    xkernel = np.outer(a = 1+np.zeros(shape = (2*eps+1)), b = range(-eps,eps+1))
    ykernel = np.outer(a = np.flip(range(-eps, eps+1)), b = 1+np.zeros(shape = (2*eps+1)))
    eu_circle = (xkernel**2+ykernel**2) <= eps*eps
    kernel[eu_circle] = 1.0
    core = convolve2d(in1 = comp, in2 = kernel, mode = "same")
    if np.sum(core)==0:
        return np.zeros_like(core)
    else:
        core = np.logical_and((core > threshold), (comp==1)).astype(int)
        return core


def Gauss_Filter(pos, neg, eps = 10.0):
    '''
    Using Gaussian Convolution to find the overlap region between two core maps. The overlapped region is the "PIL
    mask".
    :param pos: core map for the positive bitmap, should come from Cluster(pos)
    :param neg: core map for the negative bitmap, should come from Cluster(neg)
    :param eps: radius of Gaussian Filter
    :return: a weighted PIL mask
    '''

    if np.sum(pos) == 0 or np.sum(neg) == 0:
        return np.zeros_like(pos)
    else:
        #xkernel = np.outer(a=1 + np.zeros(shape=(63)), b=range(-31, 32)).astype(float)
        #ykernel = np.outer(a=np.flip(range(-31, 32)), b=1 + np.zeros(shape=(63))).astype(float)
        #kernel = np.exp(-(xkernel ** 2.0 + ykernel ** 2.0)/(2.0*100.0))

        # read the kernel from IDL
        with open("kernel.txt", "r") as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        kernel = np.zeros([63, 63])
        rowind, colind = -1, 0
        for i in range(0, len(content)):
            y = content[i]
            z = y.split(' ')
            if i % 11 == 0:
                rowind = rowind + 1
                colind = 0
            for entry in z:
                if entry != "":
                    kernel[rowind, colind] = float(entry)
                    colind = colind + 1

        kernel = kernel/np.sum(kernel)
        pos_gauss = convolve2d(in1 = pos.astype(float), in2 = kernel, mode = "same")
        neg_gauss = convolve2d(in1 = neg.astype(float), in2 = kernel, mode = "same")
        mask = pos_gauss*neg_gauss
        return mask
    
def PIL(image, threshold = 200, eps = 30, portion = 0.125):
    '''
    Generate PIL mask based on image, method follows Wang et al. (2019). This function is basically a functional
    wrapper of the functions above. Note that the final mask used by Wang et al. also checks if each pixel's value in
    conf is greater than 60 or not.
    :param image: Vertical component of the magnetic field (Br image)
    :param threshold: magnetic field threshold
    :param eps: radius deciding the boundary of "neighborhood" in DBSCAN clustering
    :param portion: portion of strong pixels within the radius needed for a point to be considered as a core point
    :return: a PIL mask
    '''

    pos, neg = Bitmap(image, threshold = threshold)
    pos_core = Cluster(comp = pos, eps = eps, portion = portion)
    neg_core = Cluster(comp = neg, eps = eps, portion = portion)
    mask = Gauss_Filter(pos = pos_core, neg = neg_core, eps = eps/3)

    return mask


