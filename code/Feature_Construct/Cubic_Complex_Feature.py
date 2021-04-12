import h5py
import numpy as np
import pandas as pd
import gudhi
from datetime import datetime
import sys

def np64_datetime(t):
    '''
    Convert numpy datetime64 object to datetime format.
    :param t: np.datetime64 timestamp
    :return: time in datetime format
    '''
    ts = (t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    t = datetime.utcfromtimestamp(ts)
    return(t)


def perseus_gen(image,fname):
    vec = np.ndarray.flatten(image)
    fileheader = "2\n" + str(image.shape[1]) + "\n" + str(image.shape[0])
    np.savetxt(fname=fname, X=vec, delimiter="\n", header=fileheader, comments="", fmt="%.3f")


def get_intensity(intsy, ret=0):
    '''
    Obtain flare intensity based on GOES dataset flare intensity records.
    :param intsy: String of flare type and flare intensity in GOES dataset, should be of format "X2.0", "M5.5", etc.
    :param ret: option of return. if set to be 0, return flare intensity (X-ray flux), otherwise return flare class
    :return: flare intensity (ret = 0), or flare class label (ret != 0)
    '''
    c = intsy[0]
    if c=="B":
        intensity = float(intsy[1:])*1e-6
    elif c=="C":
        intensity = float(intsy[1:])*1e-5
    elif c=="M":
        intensity = float(intsy[1:])*1e-4
    elif c=="X":
        intensity = float(intsy[1:])*1e-3
    else:
        intensity =  1e-8
    if ret==0:
        return intensity*1e8
    else:
        return c


def image_persistence(flare_obj, threshold, SHARP, timeline, intensity, frametime, ftype, hour, PIL=True):
    mask = np.array(flare_obj["PIL_MASK"])
    index = timeline.index(frametime[6:])
    flare_intensity = intensity[index]
    p_feature = [flare_intensity]

    for ch in SHARP:
        image = np.array(flare_obj[ch])

        # preprocess the image for cubical complex construction
        if PIL==True:
            check = np.logical_or(np.isnan(image), mask==0)
        else:
            check = np.isnan(image)

        image[check] = np.inf
        if (ch in ['Br', 'TOTUSJZ', 'TOTUSJH']):
            image = np.abs(image)

        fname = "./Penseus/" + ch + "_" + ftype + str(hour) + ".txt"
        perseus_gen(image,fname=fname)
        cubical_complex = gudhi.CubicalComplex(perseus_file=fname)
        cubical_complex.compute_persistence()
        cubical_complex.persistence()
        q = threshold[ch]

        for q_value in q.values:
            q_value = float(q_value)
            p_feature.append(cubical_complex.persistent_betti_numbers(from_value=q_value, to_value=q_value)[1]) # number
            # of live holes

    for SHARP_feature in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ','MEANJZD', 'TOTUSJZ', 'MEANALP', 'TOTUSJH', 'SAVNCPP','MEANPOT', 'MEANSHR']:
        p_feature.append(flare_obj.attrs[SHARP_feature])

    return np.array(p_feature)


def p_feature_gen(flare_type, preceding_hour, SHARP=["Br"]):
    filename = flare_type + "flare_data/" + flare_type + "_flare_" + str(preceding_hour) + "h.hdf5"
    f = h5py.File(filename, "r")
    df = pd.read_csv("Threshold.csv")
    du = pd.read_csv("GOES_dataset.csv")
    du.loc[:, 'peak_time'] = pd.to_datetime(du['peak_time'])
    du['intensity'] = du['class'].map(get_intensity)
    timeline = []
    flareclass = []
    intensity = du['intensity'].values
    K = df.shape[0]

    for index, row in du.iterrows():
        t = np64_datetime(row['peak_time'])
        flaret = datetime.strftime(t, "%Y.%m.%d_%H:%M:%S")
        timeline.append(flaret)
        flareclass.append(row['class'][0])

    allfeature = []

    for HARP in list(f.keys()):
        v = f[HARP]
        for frametime in list(v.keys()):
            flare_obj = v[frametime]
            pfeature = image_persistence(flare_obj, df, SHARP, timeline, intensity, frametime, flare_type,
                                         preceding_hour, PIL=True)
            allfeature.append(pfeature)

    colname = ["intensity"]
    for ch in SHARP:
        for j in range(K):
            colname.append(ch + str(5*(j+1)))

    for ch in ['USFLUX', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ','MEANJZD', 'TOTUSJZ', 'MEANALP', 'TOTUSJH',
               'SAVNCPP','MEANPOT', 'MEANSHR']:
        colname.append("SHARP_" + ch)

    np.save(flare_type + str(preceding_hour) + str("_TDAfeature"), np.array(allfeature))
    result = pd.DataFrame(data=np.array(allfeature), columns=colname)
    result.to_csv(flare_type + str(preceding_hour) + str("_TDAfeature") + ".csv")
    f.close()

if __name__ == "__main__":
    feature = ["Br",'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ','TOTUSJZ','TOTUSJH','MEANPOT', 'MEANSHR']
    flare = ["M","M","M","M","B","B","B","B"]
    h = [1, 6, 12, 24, 1, 6, 12, 24]
    index = int(sys.argv[1])
    p_feature_gen(flare_type=flare[index], preceding_hour=h[index], SHARP=feature)