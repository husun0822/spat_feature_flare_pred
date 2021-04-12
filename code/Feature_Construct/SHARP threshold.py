import pandas as pd
import numpy as np
import h5py
import multiprocessing as mp
import sys


def Threshold(filename, SHARP=["MEANPOT"], subsample=100):
    # This function calculates the 10%, 20%, ..., 90% quantiles for a certain SHARP parameter across all avialable data
    # the domain of the pixels being considered only include those with positive PIL weight and conf_disambig > 60.

    parameter_value = dict()
    for ch in SHARP:
        parameter_value[ch] = []

    f = h5py.File(filename, "r")

    HARPs = list(f.keys())
    for region in HARPs:
        v = f[region]
        for flare in list(v.keys()):
            mask = np.array(v[flare]['PIL_MASK'])
            conf_disambig = np.array(v[flare]['conf_disambig'])

            for image in [mask, conf_disambig]:
                image[np.isnan(image)] = 0.00000

            check = np.logical_and(mask > 0, conf_disambig > 60)
            indices = check.nonzero()
            N = indices[0].shape[0]

            if N > subsample:
                NN = np.random.choice(range(N), size=subsample, replace=False)
                indices = (indices[0][NN], indices[1][NN])


            for ch in SHARP:
                channel = np.array(v[flare][ch])
                channel[np.isnan(channel)] = 0.00000

                for param in channel[indices]:
                    parameter_value[ch].append(param)

    f.close()
    return parameter_value


def Quantile_Threshold(flarelist, hourlist, SHARP=["MEANPOT"]):
    f_handle = []

    for i in range(len(flarelist)):
        flare_type, hour = flarelist[i], hourlist[i]
        filename = flare_type + "flare_data/" + flare_type + "_flare_" + str(hour) + "h.hdf5"
        f_handle.append(filename)
        # f = h5py.File(filename, "r+")
        # f_handle.append(f)

    allparameter = dict()
    for ch in SHARP:
        allparameter[ch] = []

    for filename in f_handle:
        print(filename)
        parameter = Threshold(filename=filename, SHARP=SHARP, subsample=100)

        for ch in SHARP:
            allvalue = parameter[ch]
            for value in allvalue:
                allparameter[ch].append(value)

    allq = []

    for ch in SHARP:
        if ch in ['Br', 'TOTUSJZ', 'TOTUSJH']:
            allparameter[ch] = np.abs(np.array(allparameter[ch]))
        else:
            allparameter[ch] = np.array(allparameter[ch])
        q = np.quantile(allparameter[ch], q=np.arange(start=0.05, stop=1, step=0.05))
        allq.append(q)

    return np.array(allq)


if __name__=="__main__":
    flare = ["B", "B", "B", "B", "M", "M", "M", "M"]
    preceding_hour = [1, 6, 12, 24, 1, 6, 12, 24]
    channel = ['Br', 'MEANGAM', 'MEANGBT', 'MEANGBH', 'MEANGBZ',
               'MEANJZD', 'TOTUSJZ', 'MEANALP', 'TOTUSJH', 'SAVNCPP+', 'SAVNCPP-',
               'MEANPOT', 'MEANSHR']
    # channel = ['Br', "MEANPOT", "TOTUSJZ", "MEANSHR"]


    #quantile_threshold = []
    #columns = []

    #for feature in channel:
    #    print(feature)
    #    SHARP, q = Quantile_Threshold(flarelist=flare, hourlist=preceding_hour, SHARP=feature)
    #    columns.append(SHARP)
    #    quantile_threshold.append(q)

    quantile_threshold = Quantile_Threshold(flarelist=flare, hourlist=preceding_hour, SHARP=channel)
    np.save("quantile", quantile_threshold)
    d = pd.DataFrame(data=quantile_threshold.transpose(), columns=channel)
    d.to_csv("Threshold.csv")
