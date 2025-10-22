import numpy as np


def single_sgn_norm(sgn, normtype='maxmin'):
    if normtype == 'maxmin':
        sgn = (sgn - sgn.min()) / (sgn.max() - sgn.min())
    elif normtype == 'maxmin-1':
        sgn = (2*sgn - sgn.min() - sgn.max()) / (sgn.max() - sgn.min())
    else:
        sgn = sgn
    return sgn


def sgn_norm(sgn: np.ndarray):
    normalized_sgn = np.zeros_like(sgn)

    for i in range(sgn.size(0)):
        iq = sgn[i]
        iq = single_sgn_norm(iq, "maxmin")
        normalized_sgn[i, :, :] = iq

    return normalized_sgn


def iq_to_ap(sgn: np.ndarray):
    i = sgn[:, 0, :]
    q = sgn[:, 1, :]

    amplitude = np.sqrt(i**2 + q**2)
    phase = np.arctan2(q, i)

    ap = np.stack((amplitude, phase), axis=1)

    return ap