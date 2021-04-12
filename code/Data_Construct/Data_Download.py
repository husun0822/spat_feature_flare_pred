'''This file is used for downloading the data from JSOC.'''
import drms
import sys
import sunpy
from astropy.io import fits
from astropy.utils.data import download_file
import numpy as np
import h5py
import pandas as pd
import pickle
from datetime import datetime, timedelta

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_class_label(intsy):
    '''
    This function reads string of records of flare intensity and flare class in GOES dataset and return flare class.
    :param intsy: A string of format "[Flare Class Label][Flare Class Intensity]", e.g. M2.5, C6.0
    (this is the format of records of flare classes and intensity in GOES dataset)
    :return: flare class label: e.g. A, B, C, M, X
    '''
    flare_class = intsy[0]
    return flare_class


def np64_datetime(t):
    '''
    Convert numpy datetime64 object to datetime format.
    :param t: np.datetime64 timestamp
    :return: time in datetime format
    '''
    ts = (t - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
    t = datetime.utcfromtimestamp(ts)
    return(t)


def dload_data(GOES_path, HARP_AR_path, flare_type="M", preceding_hour=1, dtimeout=500):
    '''This function is used to download high resolution HMI image from JSOC. With pre-specified type of flare and
    number of hour preceding the flare, we end up with an hdf5 file containing the Br, Bp, Bt and conf_disambig data
    for all flares of the type at the number of hour before the flares.'''

    df = pd.read_csv(GOES_path) # read the GOES dataset
    HARP_AR_match = load_obj(HARP_AR_path) # load the HARP_AR correspondence dictionary

    filename = flare_type + "_flare_" + str(preceding_hour) + "h.hdf5" # the new hdf5 file to be constructed
    f = h5py.File(filename, mode="w")

    df['flare_class'] = df['class'].map(lambda x: x[0])
    du = df[df['flare_class']==flare_type]
    du.loc[:,'peak_time'] = pd.to_datetime(du['peak_time'])

    segs_needed = ['Br', 'Bt', 'Bp', 'conf_disambig']
    headers_needed = ['T_REC', 'rsun_ref', 'rsun_obs', 'cdelt1', 'dsun_obs']  # these are the metadata needed

    c = drms.Client()

    for index, row in du.iterrows():
        t = np64_datetime(row['peak_time'])
        flaret = datetime.strftime(t, "%Y.%m.%d_%H:%M:%S")
        t = t + timedelta(minutes=0 - t.minute % 12) - timedelta(hours=preceding_hour)
        t1 = t + timedelta(minutes=12)
        t = datetime.strftime(t, "%Y.%m.%d_%H:%M:%S") # get the time
        t1 = datetime.strftime(t1, "%Y.%m.%d_%H:%M:%S") # get the alternative time
        ar = str(row['NOAA_ar_num'])
        harp_number = "-1"

        # now search for the HARP region number
        for harp in HARP_AR_match:
            if HARP_AR_match[harp][0]==int(ar):
                harp_number = harp[4:]
                break

        if harp_number!="-1": # if we successfully found the corresponding HARP number, we could query it
            series = 'hmi.sharp_cea_720s[' + harp_number + '][' + t + '/' + '0.2h]'
            series1 = 'hmi.sharp_cea_720s[' + harp_number + '][' + t1 + '/' + '0.2h]'

            # check data availability
            si = c.info(series)
            l = si.keywords
            allkey = list(l.index.values)
            key = allkey[0]
            for j in range(1, len(allkey)):
                if j != (len(allkey) - 1):
                    key = key + ', ' + allkey[j]

            keys, segments = c.query(series, key=key, seg='Br,Bt,Bp,conf_disambig')
            if segments.shape[0]==0:
                si = c.info(series1)
                l = si.keywords
                allkey = list(l.index.values)
                key = allkey[0]
                for j in range(1, len(allkey)):
                    if j != (len(allkey) - 1):
                        key = key + ', ' + allkey[j]

                keys, segments = c.query(series1, key=key, seg='Br,Bt,Bp,conf_disambig')
                if segments.shape[0]==0:
                    continue # we skip the flare if we cannot find any data available at both time points


            grp_name = "HARP"+harp_number
            if grp_name not in list(f.keys()):
                g = f.create_group(grp_name)
            else:
                g = f[grp_name]

            flare_grp = "flare_"+flaret
            if flare_grp not in list(g.keys()):
                flare = g.create_group("flare_"+flaret)
            else:
                continue

            # write the segment data into the file
            for d in segs_needed:
                url = "http://jsoc.stanford.edu" + segments.iloc[0,].loc[d]
                data = fits.getdata(download_file(url, timeout=dtimeout))
                flare.create_dataset(d, data=data)
            for p in list(keys):
                flare.attrs[p] = keys[p].values[0]

    f.close()


if __name__ == "__main__":
    flare = ["B"]
    hours = [1,6,12,24]
    for hour in hours:
        for flaretype in flare:
            dload_data("GOES_dataset.csv","HARP_AR_matched", flare_type=flaretype, preceding_hour=hour, dtimeout=500)