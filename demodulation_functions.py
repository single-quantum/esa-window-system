import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from encoder_functions import slot_map

from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import CSM, M
from ppm_parameters import bin_length as slot_length
from ppm_parameters import (num_bins_per_symbol, symbol_length,
                            symbols_per_codeword)
from utils import flatten, moving_average


def make_time_series(time_stamps: npt.NDArray[np.float_], slot_length: float) -> npt.NDArray[np.int_]:
    """Digitize/discretize the array of time_stamps, so that it becomes a time series of zeros and ones. """
    time_vec: npt.NDArray[np.float_] = np.arange(time_stamps[0], time_stamps[-1], slot_length, dtype=float)

    # The time series vector is a vector of ones and zeros with a one if there is a pulse in that slot
    time_series: npt.NDArray[np.int_] = np.zeros(len(time_vec), dtype=np.int_)

    m: int = 0
    n: int = 0

    while m < len(time_stamps) - 1:
        if time_vec[n] <= time_stamps[m] <= time_vec[n + 1]:
            time_series[n] = 1
            m += 1
        else:
            n += 1

    return time_series


def determine_CSM_time_shift(
        csm_times: npt.NDArray[np.float_],
        time_stamps: npt.NDArray[np.float_],
        slot_length: float) -> float:
    """ Determine the time shift that is needed to shift the CSM times to the beginning of a slot.

    Because the CSM times are found with a correlation relative to a random time event,
    a time shift needs to be determined to find the true CSM time. """
    shifts = []
    csm_slot_times = np.arange(
        csm_times[0],
        csm_times[0] + num_bins_per_symbol * 20 * len(CSM) * slot_length,
        slot_length)
    n = 0
    for i in range(len(csm_slot_times) - 1):
        if csm_slot_times[i] <= time_stamps[time_stamps >= csm_times[0]][n] <= csm_slot_times[i + 1]:
            shifts.append(time_stamps[time_stamps >= csm_times[0]][n] - csm_slot_times[i])
            n += 1

    shift = np.mean(shifts)
    return shift


def find_csm_times(
        time_stamps: npt.NDArray[np.float_],
        CSM: npt.NDArray[np.int_],
        slot_length: float,
        symbol_length: float
) -> npt.NDArray[np.float_]:
    """Find the where the Codeword Synchronization Markers (CSMs) are in the sequence of `time_stamps`. """

    # + 0.5 slot length because pulse times should be in the middle of a slot.
    csm_time_stamps = np.array([slot_length * CSM[i] + i * symbol_length for i in range(len(CSM))]) + 0.5 * slot_length

    A = make_time_series(time_stamps, slot_length)
    B = make_time_series(csm_time_stamps, slot_length)

    corr: npt.NDArray[np.int_] = np.correlate(A, B, mode='valid')

    # where_corr finds the time shifts where the correlation is high enough to be a CSM.
    # Maximum correlation is 16 for 8-PPM
    # where_corr: npt.NDArray[np.int_] = np.where(corr >= 10)[0]
    where_corr = find_peaks(corr, height=9, distance=symbols_per_codeword * num_bins_per_symbol)[0]

    if where_corr.shape[0] == 0:
        raise ValueError("Could not find any CSM. ")

    # Make a moving average of the correlation to find out where the start and end is of the message
    moving_avg_corr: npt.NDArray[np.int_] = moving_average(corr, n=1000)
    message_start_idxs: npt.NDArray[np.int_] = find_peaks(
        -(moving_avg_corr - min(moving_avg_corr)) / (max(moving_avg_corr) - min(moving_avg_corr)) + 1,
        height=(0.6, 1),
        distance=symbols_per_codeword * num_bins_per_symbol)[0]

    if message_start_idxs.shape[0] == 0:
        raise ValueError("Could not find message start / end. ")

    where_csm_corr: npt.NDArray[np.int_] = where_corr[(
        where_corr >= message_start_idxs[0]) & (where_corr <= message_start_idxs[1])]

    t0: float = time_stamps[0]

    # I don't know why the -1 slot length is needed
    csm_times: npt.NDArray[np.float_] = t0 + slot_length * where_csm_corr - 1 * slot_length

    time_shift: float = determine_CSM_time_shift(csm_times, time_stamps, slot_length)
    csm_times += time_shift - 0.5 * slot_length

    return csm_times


def find_and_parse_codewords(csm_times: npt.NDArray[np.float_], peak_locations: npt.NDArray[np.float_]):
    len_codeword: int = symbols_per_codeword + len(CSM)
    len_codeword_no_CSM: int = symbols_per_codeword

    msg_symbols: list[npt.NDArray[np.float_]] = []
    num_darkcounts: int = 0
    symbols: list[float] | npt.NDArray[np.float_]

    for i in range(len(csm_times) - 1):
        start: float = csm_times[i]
        stop: float = csm_times[i + 1]
        # t0_codeword = peak_locations[start] - 0.5 * slot_length
        fraction_lost: float = (stop - start) / (symbol_length * len_codeword) - 1
        num_codewords_lost = round(fraction_lost)

        symbols, num_darkcounts = parse_ppm_symbols(
            peak_locations[peak_locations > csm_times[i]],
            csm_times[i],
            csm_times[i + 1],
            slot_length,
            symbol_length,
            num_darkcounts,
        )

        # If `parse_ppm_symbols` did not manage to parse enough symbols from the
        # peak locations, add random PPM symbols at the end of the codeword.
        if len(symbols) < len_codeword:
            diff = len_codeword - len(symbols)
            symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        if num_codewords_lost == 0 and len(symbols) > len_codeword:
            symbols = symbols[:len_codeword]

        if num_codewords_lost >= 1:
            diff = (num_codewords_lost + 1) * len_codeword_no_CSM - (len(symbols) - len(CSM))
            if diff > 0:
                symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        msg_symbols.append(np.round(symbols).astype(int))

    # Take the last CSM and parse until the end of the message.
    symbols, num_darkcounts = parse_ppm_symbols(
        peak_locations[peak_locations > csm_times[-1]],
        csm_times[-1],
        csm_times[-1] + (symbols_per_codeword + len(CSM)) * symbol_length,
        slot_length,
        symbol_length,
        num_darkcounts
    )
    msg_symbols.append(np.round(symbols).astype(int))

    print(f'Estimated number of darkcounts in message frame: {num_darkcounts}')
    return msg_symbols


def get_num_events_per_slot(csm_times, peak_locations):
    # Timespan of the entire message
    msg_end_time = csm_times[-1] + (symbols_per_codeword + len(CSM)) * symbol_length
    msg_start_idx = np.where(peak_locations >= csm_times[0])[0][0]
    msg_timespan = msg_end_time - csm_times[0]
    num_slots = int(round(msg_timespan / slot_length))

    num_events_per_slot = np.zeros(num_slots)

    for i in range(num_events_per_slot.shape[0] - 1):
        slot_start = peak_locations[msg_start_idx] + i * slot_length
        slot_end = peak_locations[msg_start_idx] + (i + 1) * slot_length

        num_events = peak_locations[(peak_locations >= slot_start) & (peak_locations < slot_end)].shape[0]
        num_events_per_slot[i] = num_events

    return num_events_per_slot


def demodulate(peak_locations: npt.NDArray) -> npt.NDArray[np.int_]:

    csm_times: npt.NDArray[np.float_] = find_csm_times(peak_locations, CSM, slot_length, symbol_length)

    msg_end_time = csm_times[-1] + (symbols_per_codeword + len(CSM)) * symbol_length
    msg_peak_locations = peak_locations[(peak_locations >= csm_times[0]) & (peak_locations <= msg_end_time)]
    # For now, this function is only used to compare results to simulations
    # events_per_slot = get_num_events_per_slot(csm_times, msg_peak_locations)

    num_detection_events: int = np.where((peak_locations >= csm_times[0]) & (
        peak_locations <= csm_times[-1] + symbols_per_codeword * symbol_length))[0].shape[0]

    print(f'Found {len(csm_times)} codewords. ')
    print(f'Number of detection events in message frame: {num_detection_events}')
    print()

    msg_symbols = find_and_parse_codewords(csm_times, peak_locations)

    print('Number of demodulated symbols: ', len(msg_symbols))

    slot_mapped_message = slot_map(flatten(msg_symbols), M)

    return slot_mapped_message
