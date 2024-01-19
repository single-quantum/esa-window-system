import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_symbol_times(
    symbol_times: npt.NDArray,
    symbol_length: float,
    slot_length: float,
    codeword_start_time: float,
    demodulated_symbols: list[float],
    num_symbols_per_codeword: int,
    start_symbol_index: int = 0,
    **kwargs
):
    """Used for debugging, this function plots the PPM symbol locations in time.

    It compares the received symbols with the expected / sent symbols. """
    with open('sent_symbols', 'rb') as f:
        sent_symbols = pickle.load(f)

    codeword_idx = kwargs.get('codeword_idx', 0)
    sent_symbols = deepcopy(sent_symbols[codeword_idx * num_symbols_per_codeword:])
    num_symbols = 6
    t0 = codeword_start_time + start_symbol_index * symbol_length
    te = t0 + num_symbols * symbol_length

    num_slots = round((te - t0) / slot_length)
    num_samples = 10 * num_slots

    symbol_start_times = np.arange(t0, te, symbol_length)
    slot_start_times = np.arange(t0, te, slot_length)
    plot_time_vector = np.arange(t0, te, (te - t0) / num_samples)

    y_received = np.zeros(plot_time_vector.shape[0])
    y_sent = np.zeros(plot_time_vector.shape[0])

    # Received symbols
    si = 0
    i = np.where(symbol_times >= plot_time_vector[0])[0][0]
    while si < num_samples - 1:
        if plot_time_vector[si] <= symbol_times[i] < plot_time_vector[si + 1]:
            y_received[si] += 1
            i += 1
        else:
            si += 1

    # Sent symbols
    sent_symbol_times = [
        sent_symbols[i] * slot_length + i * symbol_length + 0.5 * slot_length for i in range(start_symbol_index, len(sent_symbols))
    ]
    si = 0
    i = 0
    while si < num_samples - 1:
        if plot_time_vector[si] <= sent_symbol_times[i] + codeword_start_time < plot_time_vector[si + 1]:
            y_sent[si] += 1
            i += 1
        else:
            si += 1

    demodulated_symbol_times = [demodulated_symbols[i] * slot_length +
                                (i - start_symbol_index) * symbol_length for i in range(start_symbol_index, start_symbol_index + num_symbols)]

    num_slots_per_symbol = round(symbol_length / slot_length)

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(plot_time_vector - t0, y_received)
    axs[0].set_title(
        f'Received symbols (codeword = {codeword_idx+1}, symbols {start_symbol_index+1}-{start_symbol_index+num_symbols+1})')
    for t in symbol_start_times:
        axs[0].axvline(t - t0, color='red', linewidth=1)
    for i, t in enumerate(slot_start_times[:-2]):
        if (i + 2) % num_slots_per_symbol == 0:
            axs[0].axvspan(slot_start_times[i] - t0, slot_start_times[i + 2] - t0, alpha=0.4, color='grey')
        if i % num_slots_per_symbol == 0:
            continue
        axs[0].axvline(t - t0, color='gold', linewidth=1, linestyle='--')
    for i, t in enumerate(demodulated_symbol_times):
        axs[0].text(t + 0.35 * slot_length, 1.01 * y_received.max(),
                    str(int(demodulated_symbols[i + start_symbol_index])))
    axs[0].set_xticks(symbol_start_times - t0)
    axs[0].set_ylabel('Amplitude (a.u.)')

    # Reference / sent symbols
    axs[1].plot(plot_time_vector - t0, y_sent)
    axs[1].set_title('Sent symbols')
    for t in symbol_start_times:
        axs[1].axvline(t - t0, color='red', linewidth=1)
    for i, t in enumerate(slot_start_times[:-2]):
        if (i + 2) % num_slots_per_symbol == 0:
            axs[1].axvspan(slot_start_times[i] - t0, slot_start_times[i + 2] - t0, alpha=0.4, color='grey')
        if i % num_slots_per_symbol == 0:
            continue
        axs[1].axvline(t - t0, color='gold', linewidth=1, linestyle='--')
    for i, t in enumerate(sent_symbol_times[:num_symbols]):
        axs[1].text(t - start_symbol_index * symbol_length, 1.01 * y_sent.max(),
                    str(int(sent_symbols[i + start_symbol_index])))
    axs[1].set_xticks(symbol_start_times - t0)
    axs[1].set_ylabel('Amplitude (a.u.)')
    axs[1].set_xlabel('Time (s)')
    plt.show()


def find_pulses_within_symbol_frame(
    i: int,
    symbol_length: float,
    bin_times: npt.NDArray[np.float_],
    start_time: float
) -> tuple[npt.NDArray[np.float_], float, float]:
    """Find all time events (pulses) within the given symbol frame, defined by i and the symbol length.

    Returns a list of time events within the frame, as well as the symbol start and end time.
    """
    symbol_start: float = start_time + i * symbol_length
    symbol_end: float = start_time + (i + 1) * symbol_length

    symbol_frame_pulses: npt.NDArray[np.float_] = bin_times[np.logical_and(
        bin_times >= symbol_start, bin_times <= symbol_end)]

    return symbol_frame_pulses, symbol_start, symbol_end


def check_timing_requirement(pulse: float, symbol_start: float, slot_length: float) -> bool:
    """Check whether a pulse is within the CCSDS timing requirement of 10% RMS of the slot length. """
    timing_requirement: bool = True

    A: int = int((pulse - symbol_start) / slot_length)
    slot_start: float = A * slot_length
    slot_end: float = (A + 1) * slot_length
    center: float = slot_start + (slot_end - slot_start) / 2
    sigma: float = 0.1 * slot_length
    off_center_time = center - (pulse - symbol_start)

    # Pulse time does not comply with the timing requirement
    if abs(off_center_time) > sigma:
        timing_requirement = False

    return timing_requirement


def parse_ppm_symbols(
        pulse_times: npt.NDArray[np.float_],
        codeword_start_time: float,
        stop_time: float,
        slot_length: float,
        symbol_length: float,
        M: int,
        num_darkcounts: int = 0,
        **kwargs) -> tuple[list[float], int]:

    symbols: list[float] = []
    num_symbol_frames: int = int(round((stop_time - codeword_start_time) / symbol_length))
    residuals = []

    message_pulse_times = pulse_times[(pulse_times >= codeword_start_time) & (pulse_times < stop_time)]

    for i in range(num_symbol_frames):
        symbol_frame_pulses, symbol_start, _ = find_pulses_within_symbol_frame(
            i, symbol_length, message_pulse_times, codeword_start_time)

        # No symbol detected in this symbol frame
        if symbol_frame_pulses.size == 0:
            symbols.append(0)
            continue

        j = 0
        if len(symbol_frame_pulses) > 1:
            num_darkcounts += len(symbol_frame_pulses) - 1

        symbol_frame_symbols = []
        for pulse in symbol_frame_pulses:
            symbol = (pulse - symbol_start - 0.5 * slot_length) / slot_length

            # Symbols cannot be in guard slots
            if round(symbol) >= M:
                continue

            # If the symbol is too far off the bin center, it is most likely a darkcount
            # Uncomment to enforce the CCSDS timing requirement. It is commented
            # because it seems to make the error rate slightly worse.

            # timing_requirement = check_timing_requirement(pulse, symbol_start, slot_length)
            # if not timing_requirement:
            #     continue
            symbol_frame_symbols.append(symbol)
            # symbols.append(symbol)
            j += 1
            # break

        # If there were pulses detected in the symbol frame, but none of them were valid symbols, use a 0 instead.
        # This makes sure that there will always be a symbol in each symbol frame.
        if j == 0:
            symbols.append(0)
            continue
        rounded_symbols = np.round(symbol_frame_symbols)
        symbol_residuals = symbol_frame_symbols - rounded_symbols
        occurences = []
        residuals.append(np.mean(symbol_residuals))

        for symbol in np.unique(rounded_symbols):
            occurences.append(np.count_nonzero(rounded_symbols == symbol))

        best_symbol = np.unique(rounded_symbols[np.argmax(occurences)])[0]
        symbols.append(best_symbol)

    codeword_idx = kwargs.get('codeword_idx', 0)
    with open('sent_symbols', 'rb') as f:
        sent_symbols = pickle.load(f)

    num_symbol_errors = np.nonzero(
        np.round(np.array(symbols)) - sent_symbols[codeword_idx *
                                                   num_symbol_frames:(codeword_idx + 1) * num_symbol_frames]
    )[0].shape[0]
    symbol_error_ratio = num_symbol_errors / num_symbol_frames
    print(f'Codeword: {codeword_idx+1} \t symbol error ratio: {symbol_error_ratio:.3f}')

    if kwargs.get('debug_mode'):
        plot_symbol_times(pulse_times, symbol_length, slot_length,
                          codeword_start_time, symbols, num_symbol_frames, start_symbol_index=6, **kwargs)

    return symbols, num_darkcounts
