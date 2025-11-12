import pickle
import numpy as np
import scipy.io as sio
from pathlib import Path


def convert_rml_pkl2mat(pickle_path: str):
    """
    convert rml-datasets from .pkl to .mat format
    :param pickle_path: dataset path
    :return: void
    """
    Xd = pickle.load(open(pickle_path, 'rb'), encoding='latin')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbl.append((mod, snr))

    X = np.vstack(X)

    n_examples = X.shape[0]
    all_indices = range(n_examples)
    Y = list(map(lambda x: mods.index(lbl[x][0]), all_indices))
    modulations = list(map(lambda x: x[0], lbl))
    SNR = list(map(lambda x: x[1], lbl))

    # save as mat
    mat_data = {'X': X, 'Y': Y, 'modulation': modulations, 'snr': SNR}
    mat_path = str(Path(pickle_path).with_suffix(".mat"))
    sio.savemat(mat_path, mat_data)


if __name__ == "__main__":
    # convert_rml_pkl2mat("/Users/toastoffee/Documents/Datasets/RML2016.10a_dict.pkl")
    # convert_rml_pkl2mat("/Users/toastoffee/Documents/Datasets/2016.04C.multisnr.pkl")
    convert_rml_pkl2mat("/Users/toastoffee/Documents/Datasets/RML22.01A.pkl")
