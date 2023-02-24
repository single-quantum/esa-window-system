import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import copy

from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import (CSM, M, bin_length, symbol_length, symbols_per_codeword, num_bins_per_symbol)
from utils import flatten, moving_average
from scipy.signal import find_peaks

def make_time_series(time_stamps, slot_length, shift=0):
    time_vec = np.arange(time_stamps[0], time_stamps[-1], slot_length)
    A = np.zeros(len(time_vec))

    m = 0
    n = 0

    while m < len(time_stamps) - 1:
        if time_vec[n] <= time_stamps[m] <= time_vec[n + 1]:
            A[n] = 1
            m += 1
        else:
            n += 1

    return A

def determine_CSM_time_shift(csm_times, time_stamps, slot_length):
    """Because the CSM times are found with a correlation relative to a random time event, a time shift needs to be determined to find the true CSM time. """ 
    shifts = []
    csm_slot_times = np.arange(csm_times[0], csm_times[0]+num_bins_per_symbol*20*len(CSM)*slot_length, slot_length)
    n = 0
    for i in range(len(csm_slot_times)-1):
        if csm_slot_times[i] <= time_stamps[time_stamps>=csm_times[0]][n] <= csm_slot_times[i+1]:
            shifts.append(time_stamps[time_stamps>=csm_times[0]][n] - csm_slot_times[i])
            n += 1

    shift = np.mean(shifts)
    return shift

def find_csm_times(time_stamps, CSM, slot_length, symbol_length):
    """Find the where the Codeword Synchronization Markers (CSMs) are in the sequence of `time_stamps`. """

    # + 0.5 slot length because pulse times should be in the middle of a slot. 
    csm_time_stamps = np.array([slot_length*CSM[i] + i*symbol_length for i in range(len(CSM))]) + 0.5*slot_length

    A = make_time_series(time_stamps, slot_length)
    B = make_time_series(csm_time_stamps, slot_length)

    corr = np.correlate(A, B, mode='valid')

    # where_corr finds the time shifts where the correlation is high enough to be a CSM. 
    # Maximum correlation is 16 for 8-PPM
    where_corr = np.where(corr >= 10)[0]

    if where_corr.shape[0] == 0:
        raise ValueError("Could not find any CSM. ")
    
    # Make a moving average of the correlation to find out where the start and end is of the message
    moving_avg_corr = moving_average(corr, n=1000)
    message_start_idxs = find_peaks(moving_avg_corr, height=(0, 0.9), distance=symbols_per_codeword*num_bins_per_symbol)[0]
    
    if message_start_idxs.shape[0] == 0:
        raise ValueError("Could not find message start / end. ")


    where_csm_corr = copy.deepcopy(where_corr[(where_corr >= message_start_idxs[0])&(where_corr <= message_start_idxs[1])])

    t0 = time_stamps[0]

    # I don't know why the -1 slot length is needed
    csm_times = t0 + slot_length*where_csm_corr - 1*slot_length
    
    time_shift = determine_CSM_time_shift(csm_times, time_stamps, slot_length)
    csm_times += time_shift - 0.5*slot_length

    return csm_times


def find_and_parse_codewords(csm_times, peak_locations):
    len_codeword = symbols_per_codeword + len(CSM)
    len_codeword_no_CSM = symbols_per_codeword

    msg_symbols = []
    num_darkcounts = 0

    for i in range(len(csm_times) - 1):
        start = csm_times[i]
        stop = csm_times[i + 1]
        # t0_codeword = peak_locations[start] - 0.5 * bin_length
        fraction_lost = (stop - start) / (symbol_length * len_codeword) - 1
        num_codewords_lost = round(fraction_lost)

        symbols, num_darkcounts = parse_ppm_symbols(peak_locations[peak_locations>csm_times[i]], csm_times[i], csm_times[i+1], bin_length, symbol_length, num_darkcounts)

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
    symbols, num_darkcounts = parse_ppm_symbols(
        peak_locations[peak_locations>csm_times[-1]], 
        csm_times[-1], 
        csm_times[-1]+(symbols_per_codeword+len(CSM))*symbol_length, 
        bin_length, 
        symbol_length
    )
    msg_symbols.append(np.round(symbols[len(CSM):]).astype(int))

    print(f'Estimated number of darkcounts in message frame: {num_darkcounts}')
    return msg_symbols


def demodulate(peak_locations: npt.NDArray):

    csm_times = find_csm_times(peak_locations, CSM, bin_length, symbol_length)

    num_detection_events = np.where((peak_locations>=csm_times[0])&(peak_locations<=csm_times[-1]))[0].shape
    
    print(f'Found {len(csm_times)} codewords. ')
    print(f'Number of detection events in message frame: {num_detection_events}')
    print()

    msg_symbols = find_and_parse_codewords(csm_times, peak_locations)

    msg_symbols = np.array(flatten(msg_symbols))
    return msg_symbols
