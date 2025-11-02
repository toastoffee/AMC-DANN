import pickle
import numpy as np
import scipy.io as sio
from pathlib import Path


def convert_rml(pickle_path: str):

    # load pickle dataset
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    X = []  # iq
    Y = []  # labels
    SNR = []  # SNRs
    modulations = []  # modulation types

    for key in data.keys():
        # key format: (SNR, modulation) -> tuple
        modulation, snr = key
        samples = data[key]

        for i in range(samples.shape[0]):
            iq_complex = samples[i]
            iq_vec = np.stack((iq_complex[0].real, iq_complex[0].imag), axis=0)

            X.append(iq_vec)
            SNR.append(snr)
            modulations.append(modulation)

    X = np.array(X, dtype=np.float32)
    SNR = np.array(SNR, dtype=np.float32)
    modulation_to_idx = {mod: idx for idx, mod in enumerate(sorted(set(modulations)))}
    Y = np.array([modulation_to_idx[mod] for mod in modulations], dtype=np.int64)

    # save as mat
    mat_data = {'X': X, 'Y': Y, 'modulation': modulations, 'snr': SNR}
    mat_path = str(Path(pickle_path).with_suffix(".mat"))
    sio.savemat(mat_path, mat_data)


def convert_rml_ver2(pickle_path: str):
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
    # convert_rml_ver2("/Users/toastoffee/Documents/Datasets/RML2016.10a_dict.pkl")
    convert_rml_ver2("/Users/toastoffee/Documents/Datasets/2016.04C.multisnr.pkl")