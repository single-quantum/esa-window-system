import numpy as np
from numpy.random import default_rng

from parse_ppm_symbols import parse_ppm_symbols_new


def parse_ppm_symbols_single_noise_symbol():
    # With a pulse falling exactly on the slot edge, it should be parsed as a darkcount.
    symbols, _ = parse_ppm_symbols_new(np.array([0.5E-6]), 0.1E-6, 1E-6)
    return symbols


def test_parse_ppm_symbols_with_jitter():
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length
    timestamps = np.arange(0, 100E-6, 1E-6) + 0.5 * slot_length
    print(len(timestamps))
    rng = default_rng(777)
    # timestamps += rng.normal(0, 0.1 * slot_length, len(timestamps))
    symbols, _ = parse_ppm_symbols_new(timestamps, slot_length, symbol_length)
    print(len(symbols))

    assert len(symbols) == len(timestamps)


test_parse_ppm_symbols_with_jitter()
