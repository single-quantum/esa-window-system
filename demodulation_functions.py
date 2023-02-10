from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.stats import linregress
from tqdm import tqdm

import scipy as sp

from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import (CSM, M, bin_length, m, num_bins_per_symbol,
                            symbol_length, symbols_per_codeword, sample_size_awg)
from utils import flatten
from scipy.signal import correlate
from scipy.fft import fft

def check_csm(symbols, similarity_threshold=0.75):
    """Check whether the given symbols sequence is the Codeword Synchronisation Marker (CSM), by element-wise comparison. 
    
    The similarity is the number of symbols that are equal and in the same place as the CSM. """
    check_csm = False
    csm_symbols = np.round(symbols).astype(int)

    try:
        similarity = sum([1 if x == CSM[i] else 0 for (i, x) in enumerate(csm_symbols)])
    except IndexError as e:
        print(e)

    if similarity >= similarity_threshold*len(CSM):
        check_csm = True

    return check_csm


def estimate_msg_start_indexes(time_stamps, symbol_length) -> npt.NDArray:
    """Find out where the message starts, given that there is some space between messages, where only noise is received.

    Timestamps should be in seconds. """
    noise_peaks = np.where(np.diff(time_stamps) / symbol_length > 30)[0]
    where_start = np.where(np.diff(noise_peaks) > 3780)[0]

    # If only one starting index was found, use the next index as ending index.
    if where_start.shape[0] == 1:
        where_start = np.hstack((where_start, where_start[0] + 1))

    msg_start_idxs = noise_peaks[where_start]

    return msg_start_idxs

def make_time_series(time_stamps, bin_length, shift=0):
    # num_bins = int(time_stamps[-1]/bin_length)
    time_vec = np.arange(time_stamps[0], time_stamps[-1], bin_length)
    A = np.zeros(len(time_vec))

    m = 0
    n = 0
    while m < len(time_vec)-1:
        if time_vec[m] <= time_stamps[n] <= time_vec[m+1]:
            A[m] = 1
            n += 1
        m += 1

    return A

def find_csm_times(time_stamps, CSM, bin_length, symbol_length):
    """Find the where the Codeword Synchronization Markers (CSMs) are in the sequence of `time_stamps`. """

    csm_time_stamps = np.array([bin_length*CSM[i] + i*symbol_length for i in range(len(CSM))]) + 0.5*bin_length

    

    A = make_time_series(time_stamps, bin_length)
    B = make_time_series(csm_time_stamps, bin_length)

    corr = np.correlate(A, B)

    # where_corr finds all the shifts/delays in time steps of a slot_length
    where_corr = np.where(corr >= 10)[0]
    shift = 0
    while where_corr.shape[0] == 0:
        shift += 1
        A = make_time_series(time_stamps[shift:], bin_length)
        corr = np.correlate(A, B)
        where_corr = np.where(corr >= 10)[0]

        if shift > 2*len(CSM):
            raise StopIteration("Could not find CSMs")

    # where_corr += shift
    t0 = time_stamps[shift]

    csm_times = t0 + bin_length*where_corr - 1*bin_length

    return csm_times


def find_and_parse_codewords(csm_idxs, peak_locations, n0, ne):
    len_codeword = symbols_per_codeword + len(CSM)
    len_codeword_no_CSM = symbols_per_codeword

    msg_symbols = []

    for i in range(len(csm_idxs) - 1):
        start = n0 + csm_idxs[i]
        stop = n0 + csm_idxs[i + 1]
        # t0_codeword = peak_locations[start] - 0.5 * bin_length
        fraction_lost = (stop - start) / (symbol_length * len_codeword) - 1
        num_codewords_lost = round(fraction_lost)

        symbols = parse_ppm_symbols(peak_locations, csm_idxs[i], csm_idxs[i+1], bin_length, symbol_length)

        # If `parse_ppm_symbols` did not manage to parse enough symbols from the
        # peak locations, add random PPM symbols at the end of the codeword.
        if len(symbols) < len_codeword:
            diff = len_codeword - len(symbols)
            symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        if num_codewords_lost == 0 and len(symbols) > len_codeword:
            symbols = symbols[:len_codeword]
        # If there are lost CSMs, estimate where it should have been and remove these PPM symbols.
        if num_codewords_lost >= 1:
            csm_estimates_to_delete = flatten(
                [range(i * len_codeword, i * len_codeword + len(CSM)) for i in range(1, num_codewords_lost + 1)])
            symbols = np.delete(symbols, csm_estimates_to_delete)

            diff = (num_codewords_lost + 1) * len_codeword_no_CSM - (len(symbols) - len(CSM))
            if diff > 0:
                symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        msg_symbols.append(np.round(symbols[len(CSM):]).astype(int))

    # Take the last CSM and parse until the end of the message.
    symbols = parse_ppm_symbols(peak_locations, csm_idxs[i], csm_idxs[i+1], bin_length, symbol_length)
    msg_symbols.append(np.round(symbols[len(CSM):]).astype(int))

    return msg_symbols


def find_msg_indexes(peak_locations, estimated_msg_start_idxs):
    n0 = estimated_msg_start_idxs[0]
    ne = estimated_msg_start_idxs[1]

    # ne is now the start of the next message, so the exact ending position needs to be found.
    # This can be done by looking at the average symbol distance, which should be around one
    # for a message.

    j = 0
    symbol_distance = np.diff(peak_locations[n0 + j:]) / symbol_length
    while np.mean(symbol_distance[0:4]) > 3:
        j += 1
        symbol_distance = np.diff(peak_locations[n0 + j:]) / symbol_length
        if j > 15:
            raise StopIteration("Could not find msg start")

    n0 += j

    # No trimming needed
    symbol_distance = np.diff(peak_locations[n0:ne]) / symbol_length
    if np.mean(symbol_distance[-5:]) < 2:
        je = 0

        while np.mean(symbol_distance[-5:]) < 2:
            ne += 1
            symbol_distance = np.diff(peak_locations[n0:ne]) / symbol_length

            je += 1
            if je > 20000:
                raise StopIteration("Could not find msg end")

        ne -= 1
        return n0, ne

    je = 1
    while np.mean(symbol_distance[-5:]) > 2:
        ne -= 1
        symbol_distance = np.diff(peak_locations[n0:ne]) / symbol_length
        je += 1
        if je > 20000:
            raise StopIteration("Could not find msg end")

    return n0, ne


def demodulate(peak_locations: npt.NDArray):

    estimated_msg_start_idxs = estimate_msg_start_indexes(peak_locations, symbol_length)

    n0, ne = find_msg_indexes(peak_locations, estimated_msg_start_idxs)

    print(f'Number of detection events in message frame: {len(peak_locations[n0:ne])}')

    csm_times = find_csm_times(peak_locations, CSM, bin_length, symbol_length)

    print(f'Found {len(csm_times)} codewords. ')
    print()

    msg_symbols = find_and_parse_codewords(csm_times, peak_locations, n0, ne)
    # msg_symbols.append([0])

    msg_symbols = np.array(flatten(msg_symbols))
    return msg_symbols
