import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


from tqdm import tqdm

from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import CSM, num_bins_per_symbol, m


def check_csm(symbols):
    check_csm = False
    csm_symbols = np.round(symbols).astype(int)

    Es = 10
    N0 = 1

    try:
        berts_sum = sum([1 if x == CSM[i] else 0 for (i, x) in enumerate(csm_symbols)])
    except IndexError as e:
        print(e)

    if berts_sum >= 10:
        check_csm = True

    return check_csm


def find_msg_indexes(time_stamps, ASM, symbol_length) -> npt.NDArray:
    """Find out where the message starts, given that there is some space between messages, where only noise is received.

    Timestamps should be in seconds. """
    noise_peaks = np.where(np.diff(time_stamps) / symbol_length > 40)[0]
    msg_start_idxs = noise_peaks[np.where(np.diff(noise_peaks) > 3780)[0]]

    return msg_start_idxs


def find_csm_idxs(time_stamps, CSM, bin_length, symbol_length):
    csm_idxs = []

    i = 0

    while i < len(time_stamps) - len(CSM):
        symbols, _ = parse_ppm_symbols(
            time_stamps[i:i + len(CSM)] - time_stamps[i], bin_length, symbol_length)
        if len(symbols) > len(CSM):
            symbols = symbols[:len(CSM)]

        csm_found: bool = check_csm(symbols)

        if csm_found:
            print('Found CSM at idx', i)
            csm_idxs.append(i)
            i += int(0.7 * 15120 // m)

        i += 1

    return csm_idxs
