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


if __name__ == "__main__":
    # convert_rml("/Users/toastoffee/Documents/Datasets/RML2016.10a_dict.pkl")
    convert_rml("/Users/toastoffee/Documents/Datasets/2016.04C.multisnr.pkl")