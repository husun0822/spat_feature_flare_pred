import h5py
from astropy.stats import RipleysKEstimator
import numpy as np
import math
import random
import pandas as pd
from datetime import datetime
import sys
from skgstat import Variogram
import copy


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


def ptsgen(channel, PIL, threshold=200, rho=1):
    # channel must be a 2-D numpy array
    Npts = math.ceil(rho * channel.shape[0] * channel.shape[1])  # number of points to sample
    cond = np.logical_and(np.abs(channel) >= threshold, PIL!=0)
    cand = np.nonzero(cond)  # above-threshold points
    N = cand[0].shape[0]
    if N == 0:
        return -1
    w = np.abs(channel[cond])  # sample weight
    idx = random.choices(range(N), weights=w, k=Npts) # no matter how many points are above threshold,
    # we always sample Npts points
    x_coor, y_coor = [], []
    for index in idx:
        x_coor.append(cand[0][index])
        y_coor.append(cand[1][index])

    coordinate = np.array([x_coor, y_coor]).transpose()
    return coordinate


def spat_feature_gen(flare_type, preceding_hour, seed=1):
    # set random seed
    random.seed(seed)

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
    threshold_list = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]

    for HARP in list(f.keys()):
        v = f[HARP]
        for frametime in list(v.keys()):
            # print(frametime)
            flare_obj = v[frametime]
            br = np.array(flare_obj['Br'])

            # normalize br
            br[br > 5000] = 5000
            br[br < -5000] = -5000
            # br = (br+5000)/10000

            PIL = np.array(flare_obj['PIL_MASK'])
            width = br.shape[1]
            height = br.shape[0]
            rho = 500 / (width * height)  # sample at most 500 points
            index = timeline.index(frametime[6:])
            flare_intensity = intensity[index]
            fclass = flareclass[index]
            N_feature = 100 + 2  # 100-dim ripley's K feature + 2-dim Variogram feature
            N_coor = []

            for thres in threshold_list:
                # print(thres)
                coordinate = ptsgen(br, PIL, threshold=thres, rho=rho)
                if (type(coordinate)==int):
                    if thres == threshold_list[0]:
                        area = np.int(np.sum(PIL != 0))
                        feature = [flare_intensity, fclass, HARP, frametime, area, np.sum(PIL), width, height]

                        for SHARP_feature in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD',
                                              'TOTUSJZ',
                                              'MEANALP', 'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']:
                            feature.append(flare_obj.attrs[SHARP_feature])
                    for _ in range(N_feature):
                        feature.append(-1)
                    N_coor.append(0)
                else:
                    final_coordinate = []
                    z = []

                    for point in coordinate:
                        x = point[0]
                        y = point[1]
                        if (PIL[x, y] != 0):
                            final_coordinate.append([x, y])
                            z.append(br[x, y])
                    N_coor.append(len(final_coordinate))

                    if (np.sum(PIL) == 0 or len(final_coordinate) <= 10):
                        if thres == threshold_list[0]:
                            area = np.int(np.sum(PIL != 0))
                            feature = [flare_intensity, fclass, HARP, frametime, area, np.sum(PIL), width, height]

                            for SHARP_feature in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD',
                                              'TOTUSJZ',
                                              'MEANALP', 'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']:
                                feature.append(flare_obj.attrs[SHARP_feature])
                        for _ in range(N_feature):
                            feature.append(-1)
                    else:
                        final_coordinate = np.array(final_coordinate)
                        nonjitter_coordinate = copy.deepcopy(final_coordinate)
                        z = np.array(z)
                        z = z/5000
                        final_coordinate = final_coordinate + np.random.normal(loc=0, scale=1, size=final_coordinate.shape)
                        area = np.int(np.sum(PIL != 0))
                        pils = np.nonzero(PIL)
                        Kest = RipleysKEstimator(area=area, x_max=np.int(max(pils[0])), x_min=np.int(min(pils[0])),
                                             y_max=np.int(max(pils[1])), y_min=np.int(min(pils[1])))
                        r = np.linspace(0, 100, 100)
                        res = Kest(data=final_coordinate, radii=r, mode='ripley')  # Ripley's K feature

                        if thres == threshold_list[0]:
                            feature = [flare_intensity, fclass, HARP, frametime, area, np.sum(PIL), width, height]
                            for SHARP_feature in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD',
                                              'TOTUSJZ',
                                              'MEANALP', 'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']:
                                feature.append(flare_obj.attrs[SHARP_feature])
                        for kval in res:
                            feature.append(kval)

                    # variogram
                        try:
                            V = Variogram(coordinates=nonjitter_coordinate, values=z, model="exponential")
                            # res2_dist = V.data(n=50)[0]
                            # res2_var = V.data(n=50)[1]  # 100-dim Variogram feature
                            # for kval in res2_dist:
                            #    feature.append(kval)
                            # for kval in res2_var:
                            #    feature.append(kval)
                            param = V.describe()
                            feature.append(param['effective_range'])
                            feature.append(param['sill'])
                        except RuntimeError:
                            for _ in range(2):
                                feature.append(-1)

            for amt in N_coor:
                feature.append(amt)
            allfeature.append(feature)

    colname = ['intensity', 'class', 'HARP', 'Time', 'NPIL', 'areaPIL', 'width', 'height']
    for ch in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ', 'MEANALP', 'TOTUSJH',
               'SAVNCPP', 'MEANPOT', 'MEANSHR']:
        colname.append("SHARP_" + ch)
    for k in range(len(threshold_list)):
        for i in range(100):
            colname.append("Ripley" + str(k) + "_" + str(i + 1))
        for i in range(2):
            colname.append("Vario" + str(k) + "_" + str(i + 1))
    for k in range(len(threshold_list)):
        colname.append("Npts" + str(k))

    # np.save(flare_type + str(preceding_hour) + str("_spatfeature"), np.array(allfeature))
    result = pd.DataFrame(data=np.array(allfeature), columns=colname)
    result.to_csv("./spat_feature/" + flare_type + str(preceding_hour) + "_" + str(seed) + ".csv")
    f.close()


def spat_feature_gen_new(flare_type, preceding_hour, seed=1):
    # set random seed
    random.seed(seed)

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
    threshold_list = [0,200,400,600,800,1000,1200,1400,1600,1800,2000]

    for HARP in list(f.keys()):
        v = f[HARP]
        for frametime in list(v.keys()):
            # print(frametime)
            flare_obj = v[frametime]
            br = np.array(flare_obj['Br'])

            # erase extreme br values
            br[br > 5000] = 5000
            br[br < -5000] = -5000
            # br = (br+5000)/10000

            PIL = np.array(flare_obj['PIL_MASK'])
            width = br.shape[1]
            height = br.shape[0]
            rho = 500 / (width * height)  # sample at most 500 points
            index = timeline.index(frametime[6:])
            flare_intensity = intensity[index]
            fclass = flareclass[index]
            N_feature = 100 + 2  # 100-dim ripley's K feature + 2-dim Variogram feature
            N_coor = []

            for thres in threshold_list:
                # print(thres)
                coordinate = ptsgen(br, PIL, threshold=thres, rho=rho)
                if (type(coordinate)==int):
                    if thres == threshold_list[0]:
                        area = np.int(np.sum(PIL != 0))
                        feature = [flare_intensity, fclass, HARP, frametime, area, np.sum(PIL), width, height]

                        for SHARP_feature in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD',
                                              'TOTUSJZ',
                                              'MEANALP', 'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']:
                            feature.append(flare_obj.attrs[SHARP_feature])
                    for _ in range(N_feature):
                        feature.append(-1)
                    N_coor.append(0)
                else:
                    final_coordinate = []
                    z = []

                    for point in coordinate:
                        x = point[0]
                        y = point[1]
                        if (PIL[x, y] != 0):
                            final_coordinate.append([x, y])
                            z.append(br[x, y])
                    N_coor.append(len(final_coordinate))

                    if (np.sum(PIL) == 0 or len(final_coordinate) <= 10):
                        if thres == threshold_list[0]:
                            area = np.int(np.sum(PIL != 0))
                            feature = [flare_intensity, fclass, HARP, frametime, area, np.sum(PIL), width, height]

                            for SHARP_feature in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD',
                                              'TOTUSJZ',
                                              'MEANALP', 'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']:
                                feature.append(flare_obj.attrs[SHARP_feature])
                        for _ in range(N_feature):
                            feature.append(-1)
                    else:
                        final_coordinate = np.array(final_coordinate)
                        nonjitter_coordinate = copy.deepcopy(final_coordinate)
                        z = np.array(z) + np.random.normal(loc=0, scale=1, size=final_coordinate.shape[0])
                        z = z/5000 # normalize z to be within range -1 and 1
                        final_coordinate = final_coordinate + np.random.normal(loc=0, scale=1, size=final_coordinate.shape)
                        area = np.int(np.sum(PIL != 0))
                        pils = np.nonzero(PIL)
                        Kest = RipleysKEstimator(area=area, x_max=np.int(max(pils[0]))+4, x_min=np.int(min(pils[0]))-4,
                                             y_max=np.int(max(pils[1]))+4, y_min=np.int(min(pils[1]))-4)
                        r = np.linspace(0, 100, 100)
                        res = Kest(data=final_coordinate, radii=r, mode='ripley')  # Ripley's K feature

                        if thres == threshold_list[0]:
                            feature = [flare_intensity, fclass, HARP, frametime, area, np.sum(PIL), width, height]
                            for SHARP_feature in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD',
                                              'TOTUSJZ',
                                              'MEANALP', 'TOTUSJH', 'SAVNCPP', 'MEANPOT', 'MEANSHR']:
                                feature.append(flare_obj.attrs[SHARP_feature])
                        for kval in res:
                            feature.append(kval)

                    # variogram
                        try:
                            V = Variogram(coordinates=final_coordinate, values=z, model="exponential")
                            # res2_dist = V.data(n=50)[0]
                            # res2_var = V.data(n=50)[1]  # 100-dim Variogram feature
                            # for kval in res2_dist:
                            #    feature.append(kval)
                            # for kval in res2_var:
                            #    feature.append(kval)
                            param = V.describe()
                            feature.append(param['effective_range'])
                            feature.append(param['sill'])
                        except RuntimeError:
                            for _ in range(2):
                                feature.append(-1)

            for amt in N_coor:
                feature.append(amt)
            allfeature.append(feature)

    colname = ['intensity', 'class', 'HARP', 'Time', 'NPIL', 'areaPIL', 'width', 'height']
    for ch in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ', 'MEANJZD', 'TOTUSJZ', 'MEANALP', 'TOTUSJH',
               'SAVNCPP', 'MEANPOT', 'MEANSHR']:
        colname.append("SHARP_" + ch)
    for k in range(len(threshold_list)):
        for i in range(100):
            colname.append("Ripley" + str(k-1) + "_" + str(i + 1))
        for i in range(2):
            colname.append("Vario" + str(k-1) + "_" + str(i + 1))
    for k in range(len(threshold_list)):
        colname.append("Npts" + str(k-1))

    # np.save(flare_type + str(preceding_hour) + str("_spatfeature"), np.array(allfeature))
    result = pd.DataFrame(data=np.array(allfeature), columns=colname)
    result.to_csv("./spat_feature/" + flare_type + str(preceding_hour) + "_" + str(seed) + "_new.csv")
    f.close()


if __name__ == "__main__":
    flare = ["M","M","M","M","B","B","B","B"]
    h = [1, 6, 12, 24, 1, 6, 12, 24]
    index = int(sys.argv[1])
    task_index = int(int(index)//10) # index 0-79, so that task_index 0-7
    seed = int(int(index)%10) # random seed
    spat_feature_gen_new(flare_type=flare[task_index], preceding_hour=h[task_index], seed=seed)