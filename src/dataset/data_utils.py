import numpy as np
from enum import Enum


class NormType(Enum):
    MinMax = 1
    ZeroCenteredMinMax = 2


def signal_normalize(signal: np.ndarray,
                     normType: NormType = NormType.MinMax):
    """
    normalize signal in 2 ways: min-max or zero-centered-min-max
    :param signal: iq signal
    :param normType: min-max or zero-centered-min-max
    :return: normalized signal
    """
    if normType == NormType.MinMax:
        signal = (signal - signal.min()) / (signal.max() - signal.min())

    elif normType == NormType.ZeroCenteredMinMax:
        signal = (2 * signal - signal.min() - signal.max()) / (signal.max() - signal.min())

    return signal


def signals_normalize(signals: np.ndarray,
                      normType: NormType = NormType.MinMax):
    """
    normalize signals in 2 ways: min-max or zero-centered-min-max
    :param signals: iq signal
    :param normType: min-max or zero-centered-min-max
    :return: normalized signals
    """
    normalized_signals = np.zeros_like(signals)

    for i in range(signals.size(0)):
        iq = signals[i]
        iq = signal_normalize(iq, normType)
        normalized_signals[i, :, :] = iq

    return normalized_signals


def iq2ap(iq: np.ndarray):
    """
    convert signal from iq to ap
    :param iq: iq signal
    :return: signal in ap format
    """

    i = iq[:, 0, :]
    q = iq[:, 1, :]

    amplitude = np.sqrt(i**2 + q**2)
    phase = np.arctan2(q, i)
    ap = np.stack((amplitude, phase), axis=1)
    return ap

