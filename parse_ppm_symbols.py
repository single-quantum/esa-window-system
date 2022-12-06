import pickle

import matplotlib.pyplot as plt
import numpy as np
from ppm_parameters import M, m


def plot_symbol(symbol_start, symbol_end, new_symbol_start, new_symbol_end,
                bin_length, symbol_idx, num_symbols, **kwargs):
    x_data = kwargs.get('x_data')
    y_data = kwargs.get('y_data')
    t0 = kwargs.get('t0')

    start_idx = np.where((x_data >= symbol_start + t0 - bin_length) & (x_data <= symbol_start + t0))[0][0]
    stop_idx = np.where((x_data >= symbol_end + t0 - bin_length) & (x_data <= symbol_end + t0))[0][0]

    # plt.figure()
    # plt.plot(x_data[start_idx:stop_idx], y_data[start_idx:stop_idx])
    # plt.show()

    new_start_idx = np.where((x_data > new_symbol_start + t0 - bin_length) & (x_data < new_symbol_start + t0))[0][0]
    new_stop_idx = np.where((x_data > new_symbol_end + t0 - bin_length) & (x_data < new_symbol_end + t0))[0][0]
    ymax = 0.6

    symbol_frame_shift = (symbol_start + t0) * 1E3
    new_symbol_frame_shift = (new_symbol_start + t0) * 1E3

    symbol_frame_shift = t0 * 1E3
    new_symbol_frame_shift = t0 * 1E3

    fig, axs = plt.subplots(2, 1, sharey=True, figsize=(7.5, 5))
    plt.ylim(-0.05, ymax)
    axs[0].plot(x_data[start_idx:stop_idx] * 1E3 - symbol_frame_shift, y_data[start_idx:stop_idx])
    axs[1].plot(x_data[new_start_idx:new_stop_idx] * 1E3 - new_symbol_frame_shift, y_data[new_start_idx:new_stop_idx])

    axs[0].set_title('Symbol frame before timing correction')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].set_xlabel('Time relative to message start (ms)')
    axs[1].set_title('Symbol frame after timing correction')
    axs[1].set_ylabel('Voltage (V)')
    axs[1].set_xlabel('Time relative to message start (ms)')

    for i in range(16):
        axs[0].vlines((symbol_start + t0 + i * bin_length) * 1E3 - symbol_frame_shift, -
                      0.05, ymax, color='r', linestyles='--', linewidth=1)
        # axs[1].vlines((new_symbol_start+t0 + i*bin_length)*1E3-new_symbol_frame_shift, -0.05, ymax, color='r', linestyles='--', linewidth=1)
        axs[1].vlines((new_symbol_start + i * bin_length) * 1E3, -0.05, ymax, color='r', linestyles='--', linewidth=1)

    axs[0].axvspan((symbol_start + t0 + 16 * bin_length) * 1E3 - symbol_frame_shift,
                   (symbol_start + t0 + 20 * bin_length) * 1E3 - symbol_frame_shift, color='grey', alpha=0.5)
    axs[1].axvspan((new_symbol_start + t0 + 16 * bin_length) * 1E3 - new_symbol_frame_shift,
                   (new_symbol_start + t0 + 20 * bin_length) * 1E3 - new_symbol_frame_shift, color='grey', alpha=0.5)

    # axs[1].axvspan((new_symbol_start + 16*bin_length)*1E3, x_data[stop_idx]*1E3-new_symbol_frame_shift, color='grey', alpha=0.5)

    plt.suptitle(f'Symbol {symbol_idx}/{num_symbols}')
    plt.tight_layout()

    plt.show()


def parse_ppm_symbols(bin_times, bin_length, symbol_length, **kwargs):
    symbols = []

    symbol_idx = 0
    i = 0
    timing_correction = 0
    timing_corrections = []

    while i < len(bin_times) - 1:
        symbol_start = symbol_idx * symbol_length + timing_correction
        symbol_end = (symbol_idx + 1) * symbol_length + timing_correction

        # If this is the case, a symbol did not get properly sent, received or parsed.
        # Assume a 0 and continue to the next symbol, while keeping i the same.
        if bin_times[i] > symbol_end:
            symbols.append(0)
            symbol_idx += 1
            continue

        # First estimate
        symbol = (bin_times[i] - symbol_start) / bin_length


        if (0 <= (bin_times[i + 1] - symbol_start) / bin_length <= M):
            symbol = (bin_times[i + 1] - symbol_start) / bin_length
            symbols.append(symbol)
            i += 1
            symbol_idx += 1
            continue

        if bin_times[i] < symbol_start - 0.5 * bin_length or bin_times[i] > symbol_end - (M//4) * bin_length:
            # This is most likely a dark count
            i += 1
            continue


        if symbol <= -1:
            print('symbol in guard slot?')

        symbols.append(symbol)

        i += 1
        symbol_idx += 1

    return symbols, (i, symbol_idx)


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def find_ASM_from_peak_distances(ASM, inter_peak_bin_distances, num_bins_per_symbol):
    ASM_bin_distances = []
    for n, symbol in enumerate(ASM):
        ASM_bin_distances.append(n * num_bins_per_symbol + symbol)

    ASM_distances = np.diff(ASM_bin_distances)
    ASM_idxs = np.where(
        np.all(
            rolling_window(
                inter_peak_bin_distances,
                len(ASM_distances)) == ASM_distances,
            axis=1))[0]

    return ASM_idxs
