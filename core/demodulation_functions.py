import copy
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from core.encoder_functions import get_csm, slot_map
from core.parse_ppm_symbols import parse_ppm_symbols
from core.utils import flatten, moving_average


def make_time_series(time_stamps: npt.NDArray[np.float_], slot_length: float) -> npt.NDArray[np.int_]:
    """Digitize/discretize the array of time_stamps, so that it becomes a time series of zeros and ones. """
    time_vec: npt.NDArray[np.float_] = np.arange(
        time_stamps[0], time_stamps[-1] + 2 * slot_length, slot_length, dtype=float)

    # The time series vector is a vector of ones and zeros with a one if there is a pulse in that slot
    time_series: npt.NDArray[np.int_] = np.zeros(len(time_vec) - 1, dtype=np.int_)

    m: int = 0
    n: int = 0

    while m < len(time_stamps):
        if time_vec[n] <= time_stamps[m] < time_vec[n + 1]:
            time_series[n] += 1
            m += 1
        else:
            n += 1

    return time_series


def determine_CSM_time_shift(
        csm_times: npt.NDArray[np.float_],
        time_stamps: npt.NDArray[np.float_],
        slot_length: float,
        CSM: npt.NDArray[np.int_],
        num_slots_per_symbol: int) -> npt.NDArray:
    """ Determine the time shift that is needed to shift the CSM times to the beginning of a slot.

    Because the CSM times are found with a correlation relative to a random time event,
    a time shift needs to be determined to find the true CSM time. """

    csm_shifts = []

    for csm_time in csm_times:
        # for i in range(1):
        shifts = []

        csm_symbol_times = [
            csm_time + CSM[i] * slot_length + i * num_slots_per_symbol * slot_length for i in range(len(CSM))
        ]

        # Double check if copy is needed here
        # csm_timestamps = copy.deepcopy(time_stamps[time_stamps >= csm_time])
        csm_timestamps = copy.deepcopy(time_stamps[
            np.logical_and(
                time_stamps >= csm_time, time_stamps <= csm_time + num_slots_per_symbol * len(CSM) * slot_length
            )
        ])

        for ti in range(len(csm_timestamps)):
            closest_idx = (np.abs(csm_symbol_times - csm_timestamps[ti])).argmin()
            shifts.append(csm_timestamps[ti] - csm_symbol_times[closest_idx])

        # Determine z score to remove statistical outliers.
        # Perform twice in case the spread is very large (should maybe be conditioned).
        for _ in range(2):
            z_score = (np.array(shifts) - np.mean(shifts)) / np.std(shifts)
            outliers = np.where(abs(z_score) > 2)[0]
            shifts = np.delete(shifts, outliers)

        csm_shifts.append(np.mean(shifts))

    csm_shifts = np.array(csm_shifts)
    # csm_shifts = csm_shifts[0]

    return csm_shifts


def get_csm_correlation(
        time_stamps: npt.NDArray[np.float_],
        slot_length: float,
        CSM: npt.NDArray[np.int_],
        symbol_length: float,
        csm_correlation_threshold: float = 0.6,
        **kwargs):
    # + 0.5 slot length because pulse times should be in the middle of a slot.
    csm_time_stamps = np.array([slot_length * CSM[i] + i * symbol_length for i in range(len(CSM))]) + 0.5 * slot_length

    A = make_time_series(time_stamps, slot_length)
    B = make_time_series(csm_time_stamps, slot_length)

    corr: npt.NDArray = np.correlate(A, B, mode='valid')
    correlation_threshold: int = int(np.max(corr) * csm_correlation_threshold)
    if kwargs.get('debug_mode'):
        plt.figure()
        plt.plot(corr, label='CSM correlation')
        plt.axhline(correlation_threshold, color='r', linestyle='--', label='Correlation threshold')
        plt.xlabel('Shift (A.U.)')
        plt.ylabel('Correlation (-)')
        plt.title('Message correlation with the CSM')
        plt.legend(loc='lower left')
        plt.show()

    return corr


def force_peak_amount_correlation(correlation_positions, correlation_heights, correlation_heights_ordered, peak_amount):
    try:
        current_threshold = correlation_heights_ordered[int(peak_amount)-1]
    except IndexError as e:
        print(e)
        raise IndexError(e)
    chosen_peaks = np.where(correlation_heights >= current_threshold)
    return correlation_positions[chosen_peaks], current_threshold


def find_csm_times(
        time_stamps: npt.NDArray[np.float_],
        CSM: npt.NDArray[np.int_],
        slot_length: float,
        symbols_per_codeword: int,
        num_slots_per_symbol: int,
        csm_correlation,
        csm_correlation_threshold: float = 0.6,
        **kwargs
) -> npt.NDArray[np.float_]:
    """Find the where the Codeword Synchronization Markers (CSMs) are in the sequence of `time_stamps`. """

    correlation_threshold: int = int(np.max(csm_correlation) * csm_correlation_threshold)
    amount_of_slots = round((time_stamps[-1]-time_stamps[0])/slot_length)

    # where_corr finds the time shifts where the correlation is high enough to be a CSM.
    # Maximum correlation is 16 for 8-PPM
    where_corr: npt.NDArray[np.int_]
    where_corr = find_peaks(
        csm_correlation,
        height=correlation_threshold,
        distance=symbols_per_codeword * num_slots_per_symbol)[0]

    expected_number_codewords_in_data = amount_of_slots/symbols_per_codeword / num_slots_per_symbol
    where_corr_positions, where_corr_heights = find_peaks(
        csm_correlation,
        height=correlation_threshold/2,
        distance=symbols_per_codeword * num_slots_per_symbol)

    where_corr_heights = where_corr_heights['peak_heights']
    where_corr_heights_ordered = -np.sort(-where_corr_heights)
    if (len(where_corr_heights_ordered) >= int(expected_number_codewords_in_data)):
        where_corr, current_threshold = force_peak_amount_correlation(
            where_corr_positions, where_corr_heights, where_corr_heights_ordered, expected_number_codewords_in_data)
    else:
        current_threshold = correlation_threshold
    # print(current_threshold,len(where_corr))
    # There is an edge case that if the CSM appears right at the start of the timestamps,
    # that find_peaks cannot find it, even if the correlation is high enough.
    # In that case, try a simple threshold check.
    if where_corr.shape[0] == 0:
        where_corr = np.where(csm_correlation >= correlation_threshold)[0]

    if where_corr.shape[0] == 0:
        raise ValueError("Could not find any CSM. ")

    # print(where_corr)
    # Make a moving average of the correlation to find out where the start and end is of the message
    # moving_avg_corr: npt.NDArray[np.int_] = moving_average(corr, n=1000)
    moving_avg_corr: npt.NDArray[np.int_] = moving_average(csm_correlation, n=len(CSM) * num_slots_per_symbol)
    message_start_idxs: npt.NDArray[np.int_] = find_peaks(
        -(moving_avg_corr - min(moving_avg_corr)) / (max(moving_avg_corr) - min(moving_avg_corr)) + 1,
        height=(0.9, 1),
        distance=symbols_per_codeword * num_slots_per_symbol)[0]

    if message_start_idxs.shape[0] == 0:
        raise ValueError("Could not find message start / end. ")

    if len(message_start_idxs) == 1:
        expected_number_codewords_per_message = (time_stamps[-1]-time_stamps[0])/slot_length/symbols_per_codeword
    else:
        expected_number_codewords_per_message = round(
            (message_start_idxs[1]-message_start_idxs[0])/(num_slots_per_symbol*symbols_per_codeword))
    expected_number_messages = expected_number_codewords_in_data/expected_number_codewords_per_message

    message_start_postions, message_start_heights = find_peaks(
        -(moving_avg_corr - min(moving_avg_corr)) / (max(moving_avg_corr) - min(moving_avg_corr)) + 1,
        height=(0.8, 1.2),
        distance=symbols_per_codeword * num_slots_per_symbol)

    message_start_heights = message_start_heights['peak_heights']
    message_start_heights_ordered = -np.sort(-message_start_heights)

    if (len(message_start_heights_ordered) >= int(expected_number_messages)):
        message_start_idxs, current_threshold = force_peak_amount_correlation(
            message_start_postions, message_start_heights, message_start_heights_ordered, expected_number_messages)

        vals = message_start_idxs/(expected_number_codewords_per_message*num_slots_per_symbol*symbols_per_codeword)
        val = np.average(vals % 1)
        print(len(message_start_idxs))
        if (val <= (expected_number_messages % 1)):
            message_start_idxs, current_threshold = force_peak_amount_correlation(
                message_start_postions, message_start_heights, message_start_heights_ordered, expected_number_messages+1)

            print(len(message_start_idxs))

    else:
        current_threshold = (0.9, 1)

    # print(message_start_idxs)
    if kwargs.get('debug_mode'):
        fig, ax1 = plt.subplots()

        ax1.plot(csm_correlation)
        ax1.axhline(correlation_threshold, color='r', linestyle='--', label='Correlation threshold')

        ax2 = ax1.twinx()

        ax2.plot(-(moving_avg_corr - min(moving_avg_corr)) /
                 (max(moving_avg_corr) - min(moving_avg_corr)) + 1, color='tab:red')
        fig.tight_layout()
        plt.show()

    # If there is only one codeword, assume there is only one CSM, so the end
    # of the message is equal to the end of the timestamps.
    where_csm_corr: npt.NDArray[np.int_]
    where_CSM_corr_per_message = []
    if message_start_idxs.shape[0] == 1:
        where_csm_corr = where_corr[where_corr >= message_start_idxs[0]]
    else:
        message_idx: list[int]
        message_idx = kwargs.get('message_idx', [0, 1])
        where_csm_corr = where_corr[(
            where_corr >= message_start_idxs[message_idx[0]]) & (where_corr <= message_start_idxs[message_idx[1]])]
        for i in range(0, len(message_start_idxs)-1):
            where_csm_corr2: npt.NDArray[np.int_]
            where_csm_corr2 = where_corr[(
                where_corr >= message_start_idxs[i]) & (where_corr <= message_start_idxs[i+1])]
            where_CSM_corr_per_message.append(where_csm_corr2)
        where_csm_corr = where_CSM_corr_per_message[0]
        for elem in where_CSM_corr_per_message:
            if (len(elem) == expected_number_codewords_per_message):
                where_csm_corr = elem
                print('Selected one iteration')
                break

    # If where_csm_corr is empty, but where_corr is not empty, use that value for the CSM
    if where_csm_corr.shape[0] == 0 and where_corr.shape[0] != 0:
        where_csm_corr = where_corr
    if (expected_number_codewords_per_message != len(where_csm_corr)):
        print('number of codewords is different from expected')
    print(where_CSM_corr_per_message, len(where_CSM_corr_per_message))

    t0: float = time_stamps[0]
    csm_times: npt.NDArray[np.float_] = t0 + slot_length * where_csm_corr + 0.5 * slot_length

    time_shifts: npt.NDArray = determine_CSM_time_shift(csm_times, time_stamps, slot_length, CSM, num_slots_per_symbol)
    print(np.array(time_shifts) / slot_length)
    csm_times += time_shifts - 0.5 * slot_length

    return csm_times


def find_and_parse_codewords(
        csm_times: npt.NDArray[np.float_],
        pulse_timestamps: npt.NDArray[np.float_],
        CSM: npt.NDArray[np.int_],
        symbols_per_codeword: int,
        slot_length: float,
        symbol_length: float,
        M: int,
        **kwargs):
    """Using the CSM times, find and parse (demodulate) PPM codewords from the given PPM pulse timestamps. """
    len_codeword: int = symbols_per_codeword + len(CSM)
    len_codeword_no_CSM: int = symbols_per_codeword

    msg_symbols: list[npt.NDArray[np.float_]] = []
    num_darkcounts: int = 0
    symbols: list[float] | npt.NDArray[np.float_]

    for i in range(len(csm_times) - 1):
        start: float = csm_times[i]
        stop: float = csm_times[i + 1]

        fraction_lost: float = (stop - start) / (symbol_length * len_codeword) - 1
        num_codewords_lost = round(fraction_lost)

        symbols, num_darkcounts = parse_ppm_symbols(
            pulse_timestamps[pulse_timestamps > csm_times[i]],
            csm_times[i],
            csm_times[i + 1],
            slot_length,
            symbol_length,
            M,
            num_darkcounts,
            **{**kwargs, **{'codeword_idx': i}}
        )

        # If `parse_ppm_symbols` did not manage to parse enough symbols from the
        # peak locations, add random PPM symbols at the end of the codeword.
        if len(symbols) < len_codeword:
            diff = len_codeword - len(symbols)
            symbols = np.hstack((symbols, 0))

        if num_codewords_lost == 0 and len(symbols) > len_codeword:
            symbols = symbols[:len_codeword]

        if num_codewords_lost >= 1:
            diff = (num_codewords_lost + 1) * len_codeword_no_CSM - (len(symbols) - len(CSM))
            if diff > 0:
                symbols = np.hstack((symbols, 0))

        msg_symbols.append(np.round(symbols).astype(int))

    # Take the last CSM and parse until the end of the message.
    symbols, num_darkcounts = parse_ppm_symbols(
        pulse_timestamps[pulse_timestamps > csm_times[-1]],
        csm_times[-1],
        csm_times[-1] + (symbols_per_codeword + len(CSM)) * symbol_length,
        slot_length,
        symbol_length,
        M,
        num_darkcounts,
        **{**kwargs, **{'codeword_idx': len(csm_times) - 1}}
    )
    msg_symbols.append(np.round(np.array(symbols)).astype(int))

    print(f'Estimated number of darkcounts in message frame: {num_darkcounts}')
    return msg_symbols


def get_num_events_per_slot(
        csm_times,
        peak_locations: npt.NDArray[np.int_],
        CSM: npt.NDArray[np.int_],
        symbols_per_codeword: int,
        slot_length: float,
        symbol_length: float,
        M: float) -> npt.NDArray[np.int_]:
    """This function determines how many detection events there were for each slot. """

    # The factor 5/4 is determined by the protocol, which states that there
    # shall be M/4 guard slots for each PPM symbol.
    num_slots_per_codeword = int((symbols_per_codeword + len(CSM)) * 5 / 4 * M)
    num_events_per_slot = np.zeros((len(csm_times), num_slots_per_codeword))

    for i in range(len(csm_times)):
        csm_time = csm_times[i]

        # Preselect those detection peaks that are within the csm times
        if i < len(csm_times) - 1:
            message_peak_locations = peak_locations[
                (peak_locations >= csm_times[i]) & (peak_locations < csm_times[i + 1])
            ]
        else:
            message_peak_locations = peak_locations[peak_locations >= csm_times[i]]

        for j in range(num_slots_per_codeword):
            slot_start = csm_time + j * slot_length
            slot_end = csm_time + (j + 1) * slot_length

            events = message_peak_locations[
                (message_peak_locations >= slot_start) & (message_peak_locations < slot_end)
            ]

            num_events = events.shape[0]
            num_events_per_slot[i, j] = num_events

    num_events_per_slot = num_events_per_slot.flatten().astype(int)
    return num_events_per_slot


def demodulate(
    pulse_timestamps: npt.NDArray,
    M: int,
    slot_length: float,
    symbol_length: float,
    csm_correlation_threshold=0.6,
    **kwargs
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Demodulate the PPM pulse time stamps (convert the time stamps to PPM symbols).

    First, the Codeword Synchronisation Marker (CSM) is derived from the timestamps, then
    all the codewords (collection of PPM symbols) are parsed from the timestamps, for a given PPM order (M). """

    if len(pulse_timestamps) == 0:
        raise IndexError("Pulse timestamps array cannot be empty. ")

    CSM: npt.NDArray[np.int_] = get_csm(M)
    symbols_per_codeword = int(15120 / np.log2(M))
    num_slots_per_symbol = int(5 / 4 * M)

    csm_correlation = get_csm_correlation(pulse_timestamps, slot_length, CSM,
                                          symbol_length, csm_correlation_threshold=csm_correlation_threshold, **kwargs)
    csm_times: npt.NDArray[np.float_] = find_csm_times(
        pulse_timestamps, CSM, slot_length, symbols_per_codeword, num_slots_per_symbol, csm_correlation, csm_correlation_threshold=csm_correlation_threshold, **kwargs)

    # For now, this function is only used to compare results to simulations
    msg_end_time = csm_times[-1] + (symbols_per_codeword + len(CSM)) * symbol_length
    msg_pulse_timestamps = pulse_timestamps[(pulse_timestamps >= csm_times[0]) & (pulse_timestamps <= msg_end_time)]

    events_per_slot: npt.NDArray[np.int_] = get_num_events_per_slot(csm_times, msg_pulse_timestamps,
                                                                    CSM, symbols_per_codeword, slot_length, symbol_length, M)

    num_detection_events: int = np.where((pulse_timestamps >= csm_times[0]) & (
        pulse_timestamps <= csm_times[-1] + symbols_per_codeword * symbol_length))[0].shape[0]

    print(f'Found {len(csm_times)} codewords. ')
    print(f'Number of detection events in message frame: {num_detection_events}')
    print()

    msg_symbols = find_and_parse_codewords(csm_times, pulse_timestamps, CSM,
                                           symbols_per_codeword, slot_length, symbol_length, M, **kwargs)

    print('Number of demodulated symbols: ', len(flatten(msg_symbols)))

    slot_mapped_message = slot_map(flatten(msg_symbols), M)

    return slot_mapped_message, events_per_slot
