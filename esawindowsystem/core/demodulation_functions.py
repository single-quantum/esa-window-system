import copy
from math import ceil
from typing import Any
import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from esawindowsystem.core.encoder_functions import get_csm, slot_map
from esawindowsystem.core.numba_utils import get_num_events_numba
from esawindowsystem.core.parse_ppm_symbols import parse_ppm_symbols
from esawindowsystem.core.utils import flatten, moving_average
from scipy.stats import norm


def get_num_events(
        i: int,
        num_events_per_slot: npt.NDArray[np.int_],
        num_slots_per_codeword: int,
        message_peak_locations: npt.NDArray[np.float64],
        slot_starts: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:

    for j in range(message_peak_locations.shape[0]):
        idx_arr_1: npt.NDArray[np.bool] = message_peak_locations[j] >= slot_starts
        idx_arr_2: npt.NDArray[np.bool] = message_peak_locations[j] < slot_starts
        idx_arr_2 = np.roll(idx_arr_2, -1)
        # A time event should always fall into one slot and one slot only
        where_slot = np.nonzero((idx_arr_1) & (idx_arr_2))
        if where_slot[0].shape[0] > 0:
            slot_idx = where_slot[0][0]
            num_events_per_slot[i, slot_idx] += 1

    return num_events_per_slot


def make_time_series(time_stamps: npt.NDArray[np.float64],
                     slot_length: float) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    """Digitize/discretize the array of time_stamps, so that it becomes a time series of zeros and ones. """
    # Naively assume the fist timestamp is a PPM symbol.
    time_vec: npt.NDArray[np.float64] = np.arange(
        time_stamps[0], time_stamps[-1] + 50 * slot_length, slot_length, dtype=float)

    # The time series vector is a vector of ones and zeros with a one if there is a pulse in that slot
    time_series: npt.NDArray[np.int_] = np.zeros(len(time_vec) - 1, dtype=np.int_)
    deviation_from_slot_centre = np.zeros(len(time_stamps), dtype=np.float64)

    m: int = 0
    n: int = 0

    while m < len(time_stamps):
        if time_vec[n] <= time_stamps[m] < time_vec[n + 1]:
            deviation_from_slot_centre[m] = (time_vec[n+1]-0.5*slot_length) - time_stamps[m]
            m += 1
        else:
            n += 1

    # # TODO: Make sure the mean is appropriate here. -> Write test to verify.
    time_stamps += np.mean(deviation_from_slot_centre)

    # It can happen that due to this time shift, the first time stamp starts too early, in that case, skip it.
    if time_stamps[0] < time_vec[0]:
        m: int = 1
    else:
        m: int = 0

    n: int = 0
    while m < len(time_stamps):
        if time_vec[n] <= time_stamps[m] < time_vec[n + 1]:
            time_series[n] += 1
            m += 1
        else:
            n += 1

    return time_series, time_vec


def determine_CSM_time_shift(
        csm_times: npt.NDArray[np.float64],
        time_stamps: npt.NDArray[np.float64],
        slot_length: float,
        CSM: npt.NDArray[np.int_],
        num_slots_per_symbol: int) -> npt.NDArray[np.float64]:
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
        time_stamps: npt.NDArray[np.float64],
        slot_length: float,
        CSM: npt.NDArray[np.int_],
        symbol_length: float,
        csm_correlation_threshold: float = 0.6,
        **kwargs: tuple[str, Any]) -> npt.NDArray[np.int_]:
    """Discretize timestamps and return correlation of that vector with discretized CSM. """
    # + 0.5 slot length because pulse times should be in the middle of a slot.
    csm_time_stamps = np.array([slot_length * CSM[i] + i * symbol_length for i in range(len(CSM))]) + 0.0 * slot_length

    A, time_vec = make_time_series(time_stamps, slot_length)
    B, _ = make_time_series(csm_time_stamps, slot_length)

    corr: npt.NDArray[np.int_] = np.correlate(A, B, mode='valid')
    if kwargs.get('debug_mode'):
        correlation_threshold: int = int(np.max(corr) * csm_correlation_threshold)
        plt.figure()
        plt.plot(corr, label='CSM correlation')
        plt.axhline(correlation_threshold, color='r', linestyle='--', label='Correlation threshold')
        plt.xlabel('Shift (slots)', fontsize=14)
        plt.ylabel('Correlation (-)', fontsize=14)
        plt.title('Message correlation with the CSM', fontsize=16)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.legend(loc='lower left')
        plt.show()

    return corr


def force_peak_amount_correlation(
    correlation_positions: npt.NDArray[np.int_],
    correlation_heights: npt.NDArray[np.int_],
    correlation_heights_ordered: npt.NDArray[np.int_],
    peak_amount
) -> tuple[npt.NDArray[np.int_], float]:
    try:
        current_threshold = correlation_heights_ordered[int(peak_amount) - 1]
    except IndexError as e:
        print(e)
        raise IndexError(e)
    chosen_peaks = np.where(correlation_heights >= current_threshold)
    return correlation_positions[chosen_peaks], current_threshold


def find_csm_times(
        time_stamps: npt.NDArray[np.float64],
        CSM: npt.NDArray[np.int_],
        slot_length: float,
        symbols_per_codeword: int,
        num_slots_per_symbol: int,
        csm_correlation: npt.NDArray[np.int_],
        csm_correlation_threshold: float = 0.6,
        **kwargs: tuple[str, Any]
) -> npt.NDArray[np.float64]:
    """Find the where the Codeword Synchronization Markers (CSMs) are in the sequence of `time_stamps`. """

    correlation_threshold: int = int(np.max(csm_correlation) * csm_correlation_threshold)

    # where_corr finds the time shifts where the correlation is high enough to be a CSM.
    # Maximum correlation is 16 for 8-PPM
    where_corr: npt.NDArray[np.int_]
    where_corr = find_peaks(
        csm_correlation,
        height=correlation_threshold,
        distance=symbols_per_codeword * num_slots_per_symbol)[0]

    # There is an edge case that if the CSM appears right at the start of the timestamps,
    # that find_peaks cannot find it, even if the correlation is high enough.
    # In that case, try a simple threshold check.
    if where_corr.shape[0] == 0:
        where_corr = np.where(csm_correlation >= correlation_threshold)[0]

    if where_corr.shape[0] == 0:
        raise ValueError("Could not find any CSM. ")

    t0: float = time_stamps[0]
    csm_times: npt.NDArray[np.float64] = t0 + slot_length * where_corr + 0.5 * slot_length

    time_shifts: npt.NDArray = determine_CSM_time_shift(csm_times, time_stamps, slot_length, CSM, num_slots_per_symbol)
    print(f'Time shift per codeword (slot lengths): {np.array(time_shifts) / slot_length}')
    csm_times += time_shifts - 0.5 * slot_length

    return csm_times


def find_and_parse_codewords(
        csm_times: npt.NDArray[np.float64],
        pulse_timestamps: npt.NDArray[np.float64],
        CSM: npt.NDArray[np.int_],
        symbols_per_codeword: int,
        slot_length: float,
        symbol_length: float,
        M: int,
        sent_symbols: list[float] | None = None,
        **kwargs: tuple[str, Any]):
    """Using the CSM times, find and parse (demodulate) PPM codewords from the given PPM pulse timestamps. """
    len_codeword: int = symbols_per_codeword + len(CSM)
    len_codeword_no_CSM: int = symbols_per_codeword

    msg_symbols: list[npt.NDArray[np.float64]] = []
    num_darkcounts: int = 0
    symbols: list[float] | npt.NDArray[np.float64]

    codeword_idx = 0

    symbol_slot_centre_distances_list = []

    for i in range(len(csm_times) - 1):
        start: float = csm_times[i]
        stop: float = csm_times[i + 1]

        fraction_lost: float = (stop - start) / (symbol_length * len_codeword) - 1
        num_codewords_lost = round(fraction_lost)

        symbols, num_darkcounts, symbol_slot_centre_distances = parse_ppm_symbols(
            pulse_timestamps[pulse_timestamps > csm_times[i]],
            csm_times[i],
            csm_times[i + 1],
            slot_length,
            symbol_length,
            M,
            num_codewords_lost,
            sent_symbols,
            num_darkcounts,
            **{**kwargs, **{'codeword_idx': codeword_idx}}
        )

        symbol_slot_centre_distances_list.append(symbol_slot_centre_distances)
        # if num_codewords_lost >= 1:
        #     symbols = np.hstack((symbols, np.zeros(int(num_codewords_lost)*len_codeword)))

        msg_symbols.append(np.round(symbols).astype(int))

        codeword_idx += 1 + num_codewords_lost

    # Take the last CSM and parse until the end of the message.
    symbols, num_darkcounts, symbol_slot_centre_distances = parse_ppm_symbols(
        pulse_timestamps[pulse_timestamps > csm_times[-1]],
        csm_times[-1],
        csm_times[-1] + (symbols_per_codeword + len(CSM)) * symbol_length,
        slot_length,
        symbol_length,
        M,
        num_codewords_lost,
        sent_symbols,
        num_darkcounts,
        **{**kwargs, **{'codeword_idx': codeword_idx}}
    )

    symbol_slot_centre_distances_list.append(symbol_slot_centre_distances)
    msg_symbols.append(np.round(np.array(symbols)).astype(int))

    print(f'Estimated number of darkcounts in message frame: {num_darkcounts}')
    print()

    # if kwargs.get('debug_mode'):
    symbol_slot_centre_distances = flatten(symbol_slot_centre_distances_list)

    xmin = -0.5*slot_length
    xmax = 0.5*slot_length

    mean_fit, std_fit = norm.fit(symbol_slot_centre_distances)
    x_fit = np.linspace(xmin, xmax, 300)
    y_fit = norm.pdf(x_fit, mean_fit, std_fit)
    hist_bins, hist_times = np.histogram(symbol_slot_centre_distances, bins=300)
    y_max = np.max(hist_bins)
    half_max = 0.5*y_max
    where_half_max_bins = np.where(hist_bins >= half_max)[0]
    fwhm = hist_times[where_half_max_bins[-1]] - hist_times[where_half_max_bins[0]]
    fwhm_ps = fwhm*1E12

    print()
    print('System level jitter (ps)', fwhm_ps)

    if kwargs.get('debug_mode'):
        plt.figure()
        # Without density = True, the histogram and fit do not plot in the same figure
        plt.hist(symbol_slot_centre_distances, bins=300, density=True)
        plt.title(f'{M} ppm, {slot_length*1E9:.3f} ns slots')
        plt.plot(x_fit, y_fit)
        plt.text(0, 0.4*np.max(y_fit), f'STD: {std_fit*1E12:.1f} ps', fontsize=14, horizontalalignment='center')
        # plt.yticks(ticks=np.linspace(0, np.max(y_fit), 5), labels=np.linspace(0, y_max, 5))
        plt.xticks(ticks=np.linspace(xmin, xmax, 3), labels=np.round(np.linspace(xmin, xmax, 3)*1E12))
        plt.ylabel('Ocurrences')
        plt.xlabel('Time (ps)')
        plt.show()

        plt.figure()
        plt.close()

    # with open("C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\C824-02\\16 ppm\\8 ns slots\\DAC_SNSPD_correlation_calibration_message_16_ppm_80sps_660ps_pulse_width_square", "rb") as f:
    #     file_data = pickle.load(f)
    #     plt.figure()
    #     plt.plot(file_data['bin_times'], file_data['correlation_data'])
    #     plt.show()

    return msg_symbols


def get_num_events_per_slot(
        csm_times: npt.NDArray[np.float64],
        peak_locations: npt.NDArray[np.float64],
        CSM: npt.NDArray[np.int_],
        symbols_per_codeword: int,
        slot_length: float,
        M: int) -> npt.NDArray[np.int_]:
    """This function determines how many detection events there were for each slot. """

    # The factor 5/4 is determined by the protocol, which states that there
    # shall be M/4 guard slots for each PPM symbol.

    symbol_length = 5 / 4 * M * slot_length
    codeword_length = symbol_length * (symbols_per_codeword + len(CSM))
    # TODO: test if this works when the second CSM is lost.

    i: int = 0
    num_attempts: int = 0
    initial_num_csm_times: int = len(csm_times)

    while i < (len(csm_times) - 1):
        if num_attempts > 2 * initial_num_csm_times:
            # Realistically, if this process needs to be repeated more times than
            # there were CSMs to begin with, it's not going to work.
            break
        lost_codewords: list[int] = []
        expected_next_csm_time = csm_times[i] + codeword_length
        found_next_csm = (expected_next_csm_time - 0.01 *
                          codeword_length) < csm_times[i + 1] < (expected_next_csm_time + 0.01*codeword_length)
        if not found_next_csm:
            lost_codewords.append(i + 1)

            csm_times = np.insert(
                csm_times,
                lost_codewords,
                csm_times[np.array(lost_codewords) - 1] + codeword_length)

            i = 0
            num_attempts += 1

        i += 1

    # csm_times = np.insert(csm_times, [5], csm_times[4] + np.diff(csm_times)[0])

    num_slots_per_codeword = int((symbols_per_codeword + len(CSM)) * 5 / 4 * M)
    num_events_per_slot: npt.NDArray[np.int_] = np.zeros((len(csm_times), num_slots_per_codeword), dtype=np.int_)

    for i in range(len(csm_times)):
        csm_time = csm_times[i]

        # Preselect those detection peaks that are within the csm times
        if i < len(csm_times) - 1:
            message_peak_locations = peak_locations[
                (peak_locations >= csm_time) & (peak_locations < csm_times[i + 1])
            ]
        else:
            message_peak_locations = peak_locations[peak_locations >= csm_time]

        slot_starts = csm_time + np.arange(num_slots_per_codeword + 1) * slot_length

        num_events_per_slot = get_num_events(
            i, num_events_per_slot, num_slots_per_codeword, message_peak_locations, slot_starts)

    num_events_per_slot = num_events_per_slot.flatten()

    return num_events_per_slot


def demodulate(
    pulse_timestamps: npt.NDArray[np.float64],
    M: int,
    slot_length: float,
    symbol_length: float,
    sent_symbols: list[float] | None = None,
    csm_correlation_threshold: float = 0.6,
    **kwargs: dict[str, Any]
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
    csm_times: npt.NDArray[np.float64] = find_csm_times(
        pulse_timestamps, CSM, slot_length, symbols_per_codeword, num_slots_per_symbol, csm_correlation, csm_correlation_threshold=csm_correlation_threshold, **kwargs)

    # For now, this function is only used to compare results to simulations
    msg_end_time = csm_times[-1] + (symbols_per_codeword + len(CSM)) * symbol_length
    msg_pulse_timestamps = pulse_timestamps[(pulse_timestamps >= csm_times[0]) & (pulse_timestamps <= msg_end_time)]

    events_per_slot: npt.NDArray[np.int_] = get_num_events_per_slot(csm_times, msg_pulse_timestamps,
                                                                    CSM, symbols_per_codeword, slot_length, M)

    # Assuming 50% efficiency
    print()
    print('Estimated number of photons per pulse:', np.mean(events_per_slot[events_per_slot > 0])/0.5)
    print()

    num_detection_events: int = np.where((pulse_timestamps >= csm_times[0]) & (
        pulse_timestamps <= msg_end_time))[0].shape[0]

    print(f'Found {len(csm_times)} codewords. ')
    print(f'Number of detection events in message frame: {num_detection_events}')
    print()

    msg_symbols = find_and_parse_codewords(csm_times, pulse_timestamps, CSM,
                                           symbols_per_codeword, slot_length, symbol_length, M, sent_symbols, **kwargs)

    print('Number of demodulated symbols: ', len(flatten(msg_symbols)))

    slot_mapped_message = slot_map(flatten(msg_symbols), M)

    return slot_mapped_message, events_per_slot
