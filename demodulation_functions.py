import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from copy import deepcopy
from scipy.stats import linregress

from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import CSM, m, symbols_per_codeword, symbol_length, bin_length, M, num_bins_per_symbol
from utils import flatten


def check_csm(symbols):
    check_csm = False
    csm_symbols = np.round(symbols).astype(int)

    try:
        berts_sum = sum([1 if x == CSM[i] else 0 for (i, x) in enumerate(csm_symbols)])
    except IndexError as e:
        print(e)

    if berts_sum >= 12:
        check_csm = True

    return check_csm


def find_msg_indexes(time_stamps, symbol_length) -> npt.NDArray:
    """Find out where the message starts, given that there is some space between messages, where only noise is received.

    Timestamps should be in seconds. """
    noise_peaks = np.where(np.diff(time_stamps) / symbol_length > 30)[0]
    where_start = np.where(np.diff(noise_peaks) > 3780)[0]
    
    # If only one starting index was found, use the next index as ending index. 
    if where_start.shape[0] == 1:
        where_start = np.hstack((where_start, where_start[0]+1))
    
    msg_start_idxs = noise_peaks[where_start]

    return msg_start_idxs


def find_csm_idxs(time_stamps, CSM, bin_length, symbol_length):
    csm_idxs = []

    i = 0

    while i < len(time_stamps) - len(CSM):
        symbols, _ = parse_ppm_symbols(
            time_stamps[i:i + len(CSM)] - time_stamps[i], bin_length, symbol_length)
        j = 0
        while len(symbols) < len(CSM) and j < 1000:
            j += 1
            symbols, _ = parse_ppm_symbols(
                time_stamps[i:i + len(CSM) + j] - time_stamps[i], bin_length, symbol_length)

        if len(symbols) > len(CSM):
            symbols = symbols[:len(CSM)]

        csm_found: bool = check_csm(symbols)

        if csm_found:
            print('Found CSM at idx', i)
            csm_idxs.append(i)
            i = np.where(time_stamps >= time_stamps[i] + symbol_length * 15120 / m)[0][0] - 5
            # i += int(0.7 * 15120 // m)

        i += 1

    return csm_idxs

def new_method(csm_idxs, peak_locations, n0, ne):
    len_codeword = symbols_per_codeword + len(CSM)
    len_codeword_no_CSM = symbols_per_codeword

    msg_symbols = []

    for i in range(len(csm_idxs) - 1):
        start = n0 + csm_idxs[i]
        stop = n0 + csm_idxs[i + 1]
        t0_codeword = peak_locations[start]
        fraction_lost = (peak_locations[stop] - peak_locations[start]) / (symbol_length * len_codeword) - 1
        num_codewords_lost = round(fraction_lost)

        symbols, _ = parse_ppm_symbols(peak_locations[start:stop] - t0_codeword, bin_length, symbol_length)

        # If `parse_ppm_symbols` did not manage to parse enough symbols from the peak locations, add random PPM symbols at the end of the codeword. 
        if len(symbols) < len_codeword:
            diff = len_codeword - len(symbols)
            symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        # If there are lost CSMs, estimate where it should have been and remove these PPM symbols. 
        if num_codewords_lost >= 1:
            csm_estimates_to_delete = flatten(
                [range(i * len_codeword, i * len_codeword + len(CSM)) for i in range(1, num_codewords_lost + 1)])
            symbols = np.delete(symbols, csm_estimates_to_delete)

            diff = (num_codewords_lost + 1) * len_codeword_no_CSM - (len(symbols) - len(CSM))
            if diff > 1:
                symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        msg_symbols.append(np.round(symbols[len(CSM):]).astype(int))

    # Take the last CSM and parse until the end of the message. 
    t0_codeword = peak_locations[n0 + csm_idxs[-1]]
    symbols, _ = parse_ppm_symbols(peak_locations[n0 + csm_idxs[-1]:ne] - t0_codeword, bin_length, symbol_length)
    msg_symbols.append(np.round(symbols[len(CSM):]).astype(int))

    return msg_symbols

def demodulate(peak_locations: npt.NDArray):

    estimated_msg_start_idxs = find_msg_indexes(peak_locations, symbol_length)
    
    n0 = estimated_msg_start_idxs[0]
    ne = estimated_msg_start_idxs[1]

    # ne is now the start of the next message, so the exact ending position needs to be found. 
    # This can be done by looking at the average symbol distance, which should be around one 
    # for a message. 

    j = 0
    symbol_distance = np.diff(peak_locations[n0 + j:ne + j]) / symbol_length
    while np.mean(symbol_distance[0:4]) > 3:
        j += 1
        symbol_distance = np.diff(peak_locations[n0 + j:ne + j]) / symbol_length
        if j > 15:
            raise StopIteration("Could not find msg start")

    je = 1
    while np.mean(symbol_distance[-5 - je:-je]) > 1.5:
        je += 1
        if je > 20000:
            raise StopIteration("Could not find msg end")

    n0 += j
    ne -= je

    print(f'Number of detection events in message frame: {len(peak_locations[n0:ne])}')

    t0_msg = peak_locations[n0] + CSM[0] * bin_length
    t_end = peak_locations[-n0 - 1]

    csm_idxs = find_csm_idxs(peak_locations[n0:ne] - t0_msg, CSM, bin_length, symbol_length)
    
    # If 0 is not found in the CSM indexes, add it to the list of CSM indexes.
    # It should however be noted that this is not an ideal assumption, as it can be straight up wrong. 

    if csm_idxs[0] > 5:
        csm_idxs.append(0)
        csm_idxs = np.sort(csm_idxs)
        print('Zero not found in CSM indexes')

    print(f'Found {len(csm_idxs)} codewords. ')
    print()

    msg_symbols = new_method(csm_idxs, peak_locations, n0, ne)

    new = np.array(flatten(msg_symbols))

    return new