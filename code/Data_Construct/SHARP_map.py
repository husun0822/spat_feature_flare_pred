'''
This file generates the SHARP parameter mask along the polarity inversion line (PIL), based on the demand of Chip.
'''

from PIL_GEN import *
import multiprocessing as mp
import sys
import math
from scipy.signal import convolve2d
import copy
import h5py
from datetime import datetime
import os
import sys
import numpy as np
import glob
import shutil


def SHARP_prep(v):
    '''
    To get a HARP region's hdf5 file ready for SHARP*PIL calculation, we need to first calculate a few extra masks,
    such as the Bh mask, Btot mask
    :param filename:
    :return:
    '''

    framelist = list(v.keys())

    for frame in framelist:
        image = v[frame]

        # calculate the horizontal field
        if "Bh" not in list(image.keys()):
            Bx = np.array(image['Bp'])
            By = np.array(image['Bt'])
            Bh = np.sqrt(Bx * Bx + By * By)
            image.create_dataset('Bh', data = Bh)

        # calculate the total field
        if "Btot" not in list(image.keys()):
            Bx = np.array(image['Bp'])
            By = np.array(image['Bt'])
            Bz = np.array(image['Br'])
            Btot = np.sqrt(Bx * Bx + By * By + Bz * Bz)
            image.create_dataset("Btot", data = Btot)


def D_Field(field, mask):
    '''
    Function used for calculating the mask for MEANGBT, MEANGBH, MEANGBZ
    :param field:
    :param mask:
    :return:
    '''

    field = np.array(field)
    field[np.isnan(field)] = 0.0
    nx, ny = field.shape[1], field.shape[0]
    derx, dery = np.zeros_like(field), np.zeros_like(field)

    # brute force method for calculating the local gradient along x-axis and y-axis
    for i in range(1, nx - 1):
        for j in range(0, ny):
            derx[j, i] = (field[j, i + 1] - field[j, i - 1]) * 0.5

    for i in range(0, nx):
        for j in range(1, ny - 1):
            dery[j, i] = (field[j + 1, i] - field[j - 1, i]) * 0.5

    # take care of the edges
    i = 0
    for j in range(ny):
        derx[j, i] = ((-3 * field[j, i]) + (4 * field[j, i + 1]) - (field[j, i + 2])) * 0.5

    i = nx - 1
    for j in range(ny):
        derx[j, i] = ((3 * field[j, i]) + (-4 * field[j, i - 1]) - (-field[j, i - 2])) * 0.5

    j = 0
    for i in range(nx):
        dery[j, i] = ((-3 * field[j, i]) + (4 * field[j + 1, i]) - (field[(j + 2), i])) * 0.5

    j = ny - 1
    for i in range(nx):
        dery[j, i] = ((3 * field[j, i]) + (-4 * field[j - 1, i]) - (-field[j - 2, i])) * 0.5

    derivative = np.sqrt(derx * derx, dery * dery)
    #loc = np.logical_and(np.isfinite(derivative), mask!=0.0)
    derivative[np.isinf(derivative)] = 0.0
    #mean_derivative = derivative * mask/float(np.sum(loc.astype(np.int)))
    #mean_derivative = derivative * mask
    mean_derivative = derivative

    return mean_derivative


def Jz(bx, by, bz, mask, header, cdelt1):
    '''
    Calculate the z component of current.
    :param bx:
    :param by:
    :param mask:
    :return:
    '''
    bx, by, bz = np.array(bx), np.array(by), np.array(bz)
    ny, nx = bx.shape[0], bx.shape[1]
    jz = np.zeros([ny, nx])
    derx = np.zeros([ny, nx])
    dery = np.zeros([ny, nx])

    # brute force method of calculating the derivative d/dx (no consideration for edges)
    for i in range(1, nx - 1):
        for j in range(0, ny):
            derx[j, i] = (by[j, i + 1] - by[j, i - 1]) * 0.5

    # brute force method of calculating the derivative d/dy (no consideration for edges) */
    for i in range(0, nx):
        for j in range(1, ny - 1):
            dery[j, i] = (bx[j + 1, i] - bx[j - 1, i]) * 0.5

    # take care of the edges
    i = 0
    for j in range(ny):
        derx[j, i] = ((-3 * by[j, i]) + (4 * by[j, i + 1]) - (by[j, i + 2])) * 0.5

    i = nx - 1
    for j in range(ny):
        derx[j, i] = ((3 * by[j, i]) + (-4 * by[j, i - 1]) - (-by[j, i - 2])) * 0.5

    j = 0
    for i in range(nx):
        dery[j, i] = ((-3 * bx[j, i]) + (4 * bx[j + 1, i]) - (bx[(j + 2), i])) * 0.5

    j = ny - 1
    for i in range(nx):
        dery[j, i] = ((3 * bx[j, i]) + (-4 * bx[j - 1, i]) - (-bx[j - 2, i])) * 0.5

    # Calculate the sum only
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            jz[j, i] = (derx[j, i] - dery[j, i])

    munaught = 0.0000012566370614
    curl = jz * (1 / cdelt1) * (header['rsun_obs'] / header['rsun_ref']) * (0.00010) * (
            1 / munaught) * (1000.0)
    #n_pixel = np.logical_and(np.isfinite(curl), mask!=0.0)
    n_pixel = np.isfinite(curl)
    curl[np.isinf(curl)] = 0.0

    # mask for MEANJZD
    M = np.sum(n_pixel.astype(np.int))
    if M==0:
        jz_mask = np.zeros_like(bz)
    else:
        jz_mask = curl/1000

    # mask for TOTUSJZ
    #totusjz_mask = abs(mask*jz) * (cdelt1 / 1) * (header['rsun_ref'] / header['rsun_obs']) * (0.00010) * (
    #        1 / munaught)
    #totusjz_mask = abs(mask*jz)
    totusjz_mask = abs(jz)

    # mask for MEANALP
    C = ((1 / cdelt1) * (header['rsun_obs'] / header['rsun_ref']) * (1000000.0))
    #check = np.logical_and(mask!=0.0, jz!=0.0)
    check = (jz!=0.0)
    check = np.logical_and(check, bz!=0.0)
    check = np.logical_and(check, np.isinf(jz+bz)!=True)
    #bz[np.logical_not(check)] = 0.0
    bz[np.isinf(jz+bz)] = 0.0
    N = np.sum(check.astype(np.int))
    if N==0:
        meanalp_mask = np.zeros_like(bz)
    else:
        #meanalp_mask = (jz*bz*mask/N)*C # MEANJZH and ABSNJZH are similar to MEANALP,
    # so we do not produce masks for them
        #meanalp_mask = jz*bz*mask
        meanalp_mask = jz*bz

    # mask for TOTUSJH
    C1 = (1 / cdelt1) * (header['rsun_obs'] / header['rsun_ref'])
    #totusjh_mask = np.abs(jz*bz*mask)*C1
    #totusjh_mask = np.abs(jz*bz*mask)
    totusjh_mask = np.abs(jz*bz)

    # mask for SAVNCPP
    C2 = (1 / cdelt1) * (0.00010) * (1 / munaught) * (header['rsun_ref'] / header['rsun_obs'])
    #cond1 = np.logical_and(mask!=0.0, np.isfinite(bz+jz+mask))
    cond1 = np.isfinite(bz+jz)
    bz_pos = np.logical_and(bz > 0,cond1)
    bz_neg = np.logical_and(bz < 0,cond1)
    if np.sum(bz_pos)==0:
        savncpp_plus = np.zeros_like(bz)
    else:
        jz_pos = copy.deepcopy(jz)
        jz_pos[np.logical_not(bz_pos)] = 0.0
        #savncpp_plus = jz_pos*mask*C2
        #savncpp_plus = jz_pos*mask
        savncpp_plus = jz_pos

    if np.sum(bz_neg)==0:
        savncpp_minus = np.zeros_like(bz)
    else:
        jz_neg = copy.deepcopy(jz)
        jz_neg[np.logical_not(bz_neg)] = 0.0
        #savncpp_minus = jz_neg*mask*C2
        #savncpp_minus = jz_neg*mask
        savncpp_minus = jz_neg

    jz_converted = curl/1000 # the unit of jz_converted is A*m^(-2)
    return jz_mask, totusjz_mask, meanalp_mask, totusjh_mask, [savncpp_plus, savncpp_minus], jz_converted


def FreeEnergy(bx, by, bpx, bpy, mask, header, cdelt1):
    bx, by, bpx, bpy = np.array(bx), np.array(by), np.array(bpx), np.array(bpy)
    #C = (cdelt1**2) * ((header['rsun_ref'] / header['rsun_obs'])**2) * (100.0**2)
    sum1 = ((bx-bpx)**2 + (by-bpy)**2)
    #sum = sum1*C
    #n_pixel = np.logical_and(np.isfinite(sum1), mask > 0)
    n_pixel = np.isfinite(sum1)
    N = np.sum(n_pixel.astype(np.int))

    # calculate the MEANPOT mask
    if N==0:
        meanpot_mask = np.zeros_like(sum1)
    else:
        #meanpot_mask = sum1*mask/float(N*8.0*math.pi)
        #meanpot_mask = sum1*mask
        meanpot_mask = sum1

    return meanpot_mask


def Shear(bx, by, bz, bpx, bpy, mask):
    bx, by, bz, bpx, bpy = np.array(bx), np.array(by), np.array(bz), np.array(bpx), np.array(bpy)
    dotproduct = bpx*bx + bpy*by + bz**2
    magnitude_pot = np.sqrt(bpx**2 + bpy**2 + bz**2)
    magnitude_vec = np.sqrt(bx**2 + by**2 + bz**2)
    shear_angle = np.arccos(dotproduct/(magnitude_pot*magnitude_vec+0.000001))*(180.0/math.pi)
    check = np.isfinite(bx+by+bz+bpx+bpy+shear_angle)
    N = np.sum(check.astype(np.int))

    if N==0:
        meanshr_mask = np.zeros_like(bz)
        shrgt45_mask = meanshr_mask
    else:
        #shear_angle[np.logical_not(check)] = 0.0
        #meanshr_mask = shear_angle*mask/float(N)
        meanshr_mask = shear_angle
        shrgt45_mask = (shear_angle>45.0).astype(np.float)

    return meanshr_mask, shrgt45_mask


def SHARP_PIL_product(v, SHARP_param = "USFLUX"):
    # define some constants
    radsindeg = np.pi / 180.
    munaught = 0.0000012566370614

    #v = f['video0']
    #framelist = sorted(list(v.keys()), key=lambda x: x[5:], reverse=False)
    framelist = list(v.keys())

    for frame in framelist:
        image = v[frame]
        mask = np.array(image['PIL_MASK'])
        bz = np.array(image['Br'])
        bz[np.isnan(bz)] = 0.0
        bh = np.array(image['Bh'])
        bh[np.isnan(bh)] = 0.0

        bx, by, bpx, bpy = image['Bp'], image['Bt'], image['Bpx'], image['Bpy']

        # get metadata
        header = dict()
        for item in ['T_REC', 'RSUN_REF', 'RSUN_OBS', 'DSUN_OBS', 'CDELT1']:
            if item != 'T_REC':
                s = item.lower()
                header[s] = float(image.attrs[item])
            else:
                header[item] = image.attrs[item]

        cdelt1 = (math.atan((header['rsun_ref'] * header['cdelt1'] * radsindeg) / (header['dsun_obs']))) * (
                1 / radsindeg) * (3600.)

        if SHARP_param == "USFLUX":
            if "USFLUX" in list(image.keys()):
                continue
            USFLUX_mask = bz
            #USFLUX_mask = USFLUX_mask * cdelt1 * cdelt1 * (header['rsun_ref'] / header['rsun_obs']) * (
            #    header['rsun_ref'] / header['rsun_obs']) * 100.0 * 100.0
            image.create_dataset("USFLUX", data = USFLUX_mask)

        if SHARP_param == "MEANGAM":
            if "MEANGAM" in list(image.keys()):
                continue
            gam = np.arctan(bh/(np.abs(bz)+0.00001)) * 180.0/math.pi
            #gam = gam * mask
            gam[np.isinf(gam)] = 0.0
            n_pixel = np.sum((np.isinf(gam)!=True).astype(np.int))
            if n_pixel == 0:
                MEANGAM_mask = np.zeros_like(gam).astype(np.float)
            else:
                #MEANGAM_mask = gam/float(n_pixel)
                MEANGAM_mask = gam
            image.create_dataset("MEANGAM", data = MEANGAM_mask)

        if SHARP_param in ['MEANGBT', 'MEANGBH', 'MEANGBZ']:
            if SHARP_param == "MEANGBT":
                field = image['Btot']
            elif SHARP_param == "MEANGBH":
                field = image['Bh']
            else:
                field = image['Br']
            if SHARP_param in list(image.keys()):
                continue
            derivative = D_Field(field = field, mask = mask)
            image.create_dataset(SHARP_param, data = derivative)

        if SHARP_param in ['MEANJZD', 'TOTUSJZ', 'MEANALP', 'TOTUSJH', 'SAVNCPP']:
            if SHARP_param in list(image.keys()):
                continue
            if "SAVNCPP+" in list(image.keys()):
                continue
            bx, by, bz = image['Bp'], image['Bt'], image['Br']
            meanjzd, totusjz, meanalp, totusjh, savncpp, jz = Jz(bx, by, bz, mask, header = header, cdelt1 = cdelt1)
            image.create_dataset("MEANJZD", data = meanjzd)
            image.create_dataset("TOTUSJZ", data = totusjz)
            image.create_dataset("MEANALP", data = meanalp)
            image.create_dataset("TOTUSJH", data = totusjh)
            image.create_dataset("SAVNCPP+", data = savncpp[0])
            image.create_dataset("SAVNCPP-", data = savncpp[1])
            image.create_dataset("JZ", data = jz)

        if SHARP_param in ['MEANPOT', 'TOTPOT']:
            if 'MEANPOT' in list(image.keys()):
                continue
            meanpot = FreeEnergy(bx, by, bpx, bpy, mask, header = header, cdelt1 = cdelt1)
            image.create_dataset("MEANPOT", data = meanpot)

        if SHARP_param in ['MEANSHR', 'SHRGT45']:
            if SHARP_param in list(image.keys()):
                continue
            meanshr, shrgt45 = Shear(bx, by, bz, bpx, bpy, mask)
            image.create_dataset("MEANSHR", data = meanshr)
            image.create_dataset("SHRGT45", data = shrgt45)

        if SHARP_param=="R_VALUE":
            if "R_VALUE" in list(image.keys()):
                continue
            R_mask = bz*mask
            rvalue_mask = np.abs(R_mask)
            image.create_dataset("R_VALUE", data = rvalue_mask)


def SHARP_gen(v):
    SHARP_prep(v)

    # print("Generating mask for USFLUX...")
    # SHARP_PIL_product(v, SHARP_param="USFLUX")

    # print("Generating mask for MEANGAM...")
    SHARP_PIL_product(v, SHARP_param="MEANGAM")

    # print("Generating mask for MEANGBT...")
    SHARP_PIL_product(v, SHARP_param="MEANGBT")

    # print("Generating mask for MEANGBH...")
    SHARP_PIL_product(v, SHARP_param="MEANGBH")

    # print("Generating mask for MEANGBZ...")
    SHARP_PIL_product(v, SHARP_param="MEANGBZ")

    # print("Generating masks for MEANJZD, TOTUSJZ, MEANALP, TOTUSJH, SAVNCPP...")
    SHARP_PIL_product(v, SHARP_param="MEANJZD")

    # print("Generating mask for MEANPOT...")
    SHARP_PIL_product(v, SHARP_param="MEANPOT")

    # print("Generating masks for MEANSHR, SHRGT45...")
    SHARP_PIL_product(v, SHARP_param="MEANSHR")


def AllMASK_Gen(flare_type, preceding_hour):
    filename = flare_type+"flare_data/" + flare_type + "_flare_" + str(preceding_hour) + "h.hdf5"
    f = h5py.File(filename, "r+")
    allharp = list(f.keys())

    #pool = mp.Pool(processes=8)
    #process_input = []

    #for v in allharp:
    #    process_input.append([f[v]])

    #result = [pool.apply_async(SHARP_gen, t) for t in process_input]
    #output = [p.get() for p in result]

    print("Process starts!")

    for harp in allharp:
        SHARP_gen(f[harp])

    print("ALL MASKS COMPLETED!")

    f.close()



if __name__ == "__main__":
    flare = ["B", "B", "B", "B", "M", "M", "M", "M"]
    h = [1, 6, 12, 24, 1, 6, 12, 24]

    index = int(sys.argv[1])

    # flare = ["M"]
    # h = [1]
    # index = 0
    AllMASK_Gen(flare_type=flare[index], preceding_hour=h[index])