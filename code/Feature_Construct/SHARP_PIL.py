import h5py
import numpy as np
import math
import pandas as pd
from datetime import datetime
import sys


# miscellaneous functions
def get_intensity(intsy, ret=0):
    '''
    Obtain flare intensity based on GOES dataset flare intensity records.
    :param intsy: String of flare type and flare intensity in GOES dataset, should be of format "X2.0", "M5.5", etc.
    :param ret: option of return. if set to be 0, return flare intensity (X-ray flux), otherwise return flare class
    :return: flare intensity (ret = 0), or flare class label (ret != 0)
    '''
    c = intsy[0]
    if c == "B":
        intensity = float(intsy[1:]) * 1e-6
    elif c == "C":
        intensity = float(intsy[1:]) * 1e-5
    elif c == "M":
        intensity = float(intsy[1:]) * 1e-4
    elif c == "X":
        intensity = float(intsy[1:]) * 1e-3
    else:
        intensity = 1e-8
    if ret == 0:
        return intensity * 1e8
    else:
        return c


def np64_datetime(t):
    '''
    Convert numpy datetime64 object to datetime format.
    :param t: np.datetime64 timestamp
    :return: time in datetime format
    '''
    ts = (t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    t = datetime.utcfromtimestamp(ts)
    return (t)


def sharp_feature_gen(flare_type, preceding_hour):
    filename = flare_type + "flare_data/" + flare_type + "_flare_" + str(preceding_hour) + "h.hdf5"
    f = h5py.File(filename, "r")

    du = pd.read_csv("GOES_dataset.csv")
    du.loc[:, 'peak_time'] = pd.to_datetime(du['peak_time'])
    du['intensity'] = du['class'].map(get_intensity)
    timeline = []
    flareclass = []
    intensity = du['intensity'].values

    for index, row in du.iterrows():
        t = np64_datetime(row['peak_time'])
        flaret = datetime.strftime(t, "%Y.%m.%d_%H:%M:%S")
        timeline.append(flaret)
        flareclass.append(row['class'][0])

    allfeature = []

    for HARP in list(f.keys()):
        v = f[HARP]
        for frametime in list(v.keys()):
            # print(frametime)
            flare_obj = v[frametime]
            br = np.array(flare_obj['Br'])
            PIL = np.array(flare_obj['PIL_MASK'])
            width = br.shape[1]
            height = br.shape[0]

            index = timeline.index(frametime[6:])
            flare_intensity = intensity[index]
            fclass = flareclass[index]
            feature = [flare_intensity, fclass, HARP, frametime]

            # calculate ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH',
            # 'MEANGBZ', 'MEANJZD','TOTUSJZ', 'MEANALP',
            # 'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']
            # get the header metadata
            radsindeg = np.pi / 180.
            header = dict()
            for item in ['T_REC', 'RSUN_REF', 'RSUN_OBS', 'DSUN_OBS', 'CDELT1']:
                if item != 'T_REC':
                    s = item.lower()
                    header[s] = float(flare_obj.attrs[item])
                else:
                    header[item] = flare_obj.attrs[item]

            cdelt1 = (math.atan((header['rsun_ref'] * header['cdelt1'] * radsindeg) / (header['dsun_obs']))) * (
                    1 / radsindeg) * (3600.)

            # USFLUX
            br[np.isnan(br)] = 0.0
            PIL[np.isnan(PIL)] = 0.0
            USFLUX = np.sum(np.abs(br*PIL)) * cdelt1 * cdelt1 * (header['rsun_ref'] / header['rsun_obs']) * (
                header['rsun_ref'] / header['rsun_obs']) * 100.0 * 100.0

            # MEANGAM
            MEANGAM = np.array(flare_obj['MEANGAM'])
            check = np.logical_and(np.isfinite(MEANGAM), PIL!=0)
            x = MEANGAM * PIL
            if (np.sum(check)==0):
                MEANGAM = 0.0
            else:
                MEANGAM = np.sum(x[check])/np.sum(check)

            # MEANGBT
            MEANGBT = np.array(flare_obj['MEANGBT'])
            check = np.logical_and(np.isfinite(MEANGBT), PIL != 0)
            x = MEANGBT * PIL
            if (np.sum(check)==0):
                MEANGBT = 0.0
            else:
                MEANGBT = np.sum(x[check])/np.sum(check)

            # MEANGBH
            MEANGBH = np.array(flare_obj['MEANGBH'])
            check = np.logical_and(np.isfinite(MEANGBH), PIL != 0)
            x = MEANGBH * PIL
            if (np.sum(check) == 0):
                MEANGBH = 0.0
            else:
                MEANGBH = np.sum(x[check]) / np.sum(check)

            # MEANGBZ
            MEANGBZ = np.array(flare_obj['MEANGBZ'])
            check = np.logical_and(np.isfinite(MEANGBZ), PIL != 0)
            x = MEANGBZ * PIL
            if (np.sum(check)==0):
                MEANGBZ = 0.0
            else:
                MEANGBZ = np.sum(x[check])/np.sum(check)

            # MEANJZD
            curl = np.array(flare_obj['MEANJZD']) * 1000
            check = np.logical_and(np.isfinite(curl), PIL != 0)
            x = PIL * curl
            if (np.sum(check)==0):
                MEANJZD = 0.0
            else:
                MEANJZD = np.sum(x[check])/np.sum(check)

            # TOTUSJZ
            absjz = np.array(flare_obj['TOTUSJZ'])
            absjz[np.isnan(absjz)] = 0.0
            munaught = 0.0000012566370614
            TOTUSJZ = np.sum(PIL * absjz * (cdelt1 / 1) * (header['rsun_ref'] / header['rsun_obs'])
                             * (0.00010) * (1 / munaught))

            # MEANALP
            MEANALP = np.array(flare_obj['MEANALP'])
            check = np.logical_and(absjz!=0, br!=0)
            check = np.logical_and(check, PIL!=0)
            check = np.logical_and(check, np.isfinite(absjz + br + PIL))
            x = PIL * MEANALP
            C = ((1 / cdelt1) * (header['rsun_obs'] / header['rsun_ref']) * (1000000.0))
            if (np.sum(check)==0):
                MEANALP = 0.0
            else:
                MEANALP = C * np.sum(x[check])/np.sum(check)

            # TOTUSJH
            TOTUSJH = np.array(flare_obj['TOTUSJH'])
            TOTUSJH[np.isnan(TOTUSJH)] = 0.0
            x = PIL * TOTUSJH
            TOTUSJH = np.sum(x) * (1 / cdelt1) * (header['rsun_obs'] / header['rsun_ref'])

            # SAVNCPP
            C2 = (1 / cdelt1) * (0.00010) * (1 / munaught) * (header['rsun_ref'] / header['rsun_obs'])
            savncpp1 = np.array(flare_obj['SAVNCPP+'])
            savncpp2 = np.array(flare_obj['SAVNCPP-'])
            SAVNCPP = np.abs(np.sum(PIL * savncpp1) * C2) + np.abs(np.sum(PIL * savncpp2) * C2)

            # MEANPOT
            sum1 = np.array(flare_obj['MEANPOT'])
            check = np.logical_and(np.isfinite(sum1),PIL!=0)
            x = PIL * sum1
            if (np.sum(check)==0):
                MEANPOT = 0.0
            else:
                MEANPOT = np.sum(x[check])/(np.sum(check) * 8.0 * math.pi)

            # MEANSHR
            shear = np.array(flare_obj['MEANSHR'])
            bx, by, bz, bpx, bpy = np.array(flare_obj['Bp']), np.array(flare_obj['Bt']), np.array(flare_obj['Br']), \
                                   np.array(flare_obj['Bpx']), np.array(flare_obj['Bpy'])
            check = np.logical_and(np.isfinite(bx+by+bz+bpx+bpy+shear+PIL), PIL!=0)
            x = PIL * shear
            if (np.sum(check)==0):
                MEANSHR = 0.0
            else:
                MEANSHR = np.sum(x[check])/np.sum(check)

            for item in [USFLUX, MEANGAM, MEANGBH, MEANGBT, MEANGBZ, MEANJZD, TOTUSJZ, MEANALP, TOTUSJH, SAVNCPP, MEANPOT,
                 MEANSHR]:
                feature.append(item)
            allfeature.append(feature)

    colname = ['intensity', 'class', 'HARP', 'Time', 'USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH',
            'MEANGBZ', 'MEANJZD','TOTUSJZ', 'MEANALP',
            'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']

    # np.save(flare_type + str(preceding_hour) + str("_spatfeature"), np.array(allfeature))
    result = pd.DataFrame(data=np.array(allfeature), columns=colname)
    result.to_csv("./spat_feature/" + flare_type + str(preceding_hour) + "_" + "SHARP_PIL.csv")
    f.close()


if __name__ == "__main__":
    flare = ["M","M","M","M","B","B","B","B"]
    h = [1, 6, 12, 24, 1, 6, 12, 24]
    index = int(sys.argv[1])
    sharp_feature_gen(flare_type=flare[index], preceding_hour=h[index])