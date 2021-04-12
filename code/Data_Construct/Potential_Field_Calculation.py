import h5py
import numpy as np
import multiprocessing as mp
from scipy.signal import convolve2d
import sys

def greenpot(bz, HARP_region, flare):
    # print('Calculating the potential field. This takes a minute.')
    nx, ny = bz.shape[1], bz.shape[0]

    # define the monopole depth, dz
    dz = 0.001

    # bztmp = bz
    # bztmp[np.isnan(bz)] = 0.0

    #iwindow = 0
    #if (nnx > nny):
    #    iwindow = nnx
    #else:
    #    iwindow = nny

    #rwindow = float(iwindow)
    #rwindow = rwindow * rwindow + 0.01  # must be square
    #rwindow = 1.0e2  # limit the window size to be 10.
    #rwindow = np.sqrt(rwindow)
    #iwindow = int(rwindow)
    iwindow = 10

    rd1, rd2 = np.meshgrid(range(-iwindow, iwindow + 1), range(-iwindow, iwindow + 1))
    rdist_new = rd1 * rd1 + rd2 * rd2 + dz ** 2
    rdist_final = 1.0 / np.sqrt(rdist_new)
    pfpot_final = convolve2d(in1=bz, in2=rdist_final, mode="same") * dz
    pfpot_final[np.isnan(bz)] = 0.0

    filter_x = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]]) * (-0.5)
    filter_y = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]) * (-0.5)
    bpx = convolve2d(in1=pfpot_final, in2=filter_x, mode="valid")
    bpx = np.pad(bpx, (1, 1), 'constant', constant_values=0)

    bpy = convolve2d(in1=pfpot_final, in2=filter_y, mode="valid")
    bpy = np.pad(bpy, (1, 1), 'constant', constant_values=0)

    return bpx, bpy, HARP_region, flare


def pot_field_prep(flare_type, preceding_hour, core = 8):
    filename = flare_type + "flare_data/" + flare_type + "_flare_" + str(preceding_hour) + "h.hdf5"
    f = h5py.File(filename, mode="r+")

    pool = mp.Pool(processes=core)
    process_input = []

    for HARP_region in list(f.keys()):
        g = f[HARP_region]
        for flare in list(g.keys()):
            if "Bpx" in g[flare].keys():
                continue
            else:
                bz = np.array(g[flare]['Br'])
                process_input.append([bz, HARP_region, flare])

    results = [pool.apply_async(greenpot, t) for t in process_input]
    output = [p.get() for p in results]

    for item in output:
        HARP_region = item[2]
        flare = item[3]
        g = f[HARP_region]
        frame = g[flare]
        frame.create_dataset("Bpx", data = item[0])
        frame.create_dataset("Bpy", data = item[1])


if __name__=="__main__":
    flare = ["B", "B", "B", "B", "M", "M", "M", "M"]
    h = [1, 6, 12, 24, 1, 6, 12, 24]

    index = int(sys.argv[1])
    pot_field_prep(flare_type=flare[index], preceding_hour=h[index])