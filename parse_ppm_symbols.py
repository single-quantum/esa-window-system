import numpy as np

from ppm_parameters import M


def check_darkcount(bin_time, symbol_start, symbol_end, bin_length, symbol):
    guard_slot_darkcount = False
    jitter_darkcount = False
    # Then, check if the bin time falls within the 10% RMS slot width requirement
    time_offset = (symbol - round(symbol)) * bin_length
    sigma = 0.1 * bin_length
    if abs(time_offset) > 3*sigma:
        jitter_darkcount = True
        return guard_slot_darkcount, jitter_darkcount

    # First, check if the symbol was present in a guard slot
    if bin_time < symbol_start - 0.5 * bin_length or bin_time > symbol_end - (M // 4) * bin_length:
        guard_slot_darkcount = True
        return guard_slot_darkcount, jitter_darkcount

    return guard_slot_darkcount, jitter_darkcount


def parse_ppm_symbols(bin_times, bin_length, symbol_length, **kwargs):
    symbols = []

    symbol_idx = 0
    i = 0
    guard_slot_darkcounts = 0
    jitter_darkcounts = 0

    while i < len(bin_times):
        symbol_start = symbol_idx * symbol_length
        symbol_end = (symbol_idx + 1) * symbol_length

        # If this is the case, a symbol did not get properly sent, received or parsed.
        # Assume a 0 and continue to the next symbol, while keeping i the same.
        if bin_times[i] > symbol_end:
            symbols.append(0)
            symbol_idx += 1
            continue

        # First estimate
        symbol = (bin_times[i] - symbol_start) / bin_length

        guard_slot_darkcount, jitter_darkcount = check_darkcount(
            bin_times[i], symbol_start, symbol_end, bin_length, symbol)

        if guard_slot_darkcount:
            guard_slot_darkcounts += 1
            i += 1
            continue

        if jitter_darkcount:
            jitter_darkcounts += 1
            i += 1
            continue

        if i < len(bin_times) - 1 and (0 <= (bin_times[i + 1] - symbol_start) / bin_length <= M):
            symbol = (bin_times[i + 1] - symbol_start) / bin_length
            symbols.append(symbol)
            i += 1
            symbol_idx += 1
            continue

        if symbol <= -1:
            print('symbol in guard slot?')

        symbols.append(symbol)

        i += 1
        symbol_idx += 1

    if len(bin_times) > 100:
        print('jitter darkcounts', jitter_darkcounts, i, symbol_idx)
    return symbols, (i, symbol_idx)


def rolling_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a. strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
