import numpy as np
import pytest
from numpy.random import default_rng

from demodulation_functions import demodulate
from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import CSM, M, num_bins_per_symbol


def test_parse_ppm_symbols_single_noise_symbol():
    # With a pulse falling in a guard slot, it will be parsed as a 0.
    slot_length = 0.1E-6
    symbol_length = 1E-6
    symbols, _ = parse_ppm_symbols(np.array([symbol_length - slot_length]), 0, 1E-6, slot_length, symbol_length)
    assert len(symbols) == 1
    assert symbols[0] == 0


def test_parse_ppm_symbols_single_symbol():
    symbols, _ = parse_ppm_symbols(np.array([0.55E-6]), 0, 1E-6, 0.1E-6, 1E-6)
    assert len(symbols) == 1
    # Approx is needed due to floating point representation
    assert symbols[0] == pytest.approx(5)


def test_parse_ppm_symbols_multiple_symbols():
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length
    num_symbols = 100

    timestamps = np.linspace(0, (num_symbols - 1) * symbol_length, num_symbols) + 0.5 * slot_length
    symbols, _ = parse_ppm_symbols(timestamps, 0, timestamps[-1] + symbol_length, slot_length, symbol_length)
    symbols = np.round(symbols).astype(int)
    print(symbols)

    assert len(symbols) == len(timestamps)
    assert np.all(symbols == 0)


def test_parse_ppm_symbols_multiple_symbols_with_jitter():
    """At this time there is no check for the timing requirement, so no symbols should be dropped. """
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length
    num_symbols = 10000

    timestamps = np.arange(0, num_symbols * symbol_length, 1E-6) + 0.5 * slot_length + slot_length
    rng = default_rng(777)
    timestamps += rng.normal(0, 0.1 * slot_length, len(timestamps))

    symbols, _ = parse_ppm_symbols(timestamps, 0, timestamps[-1] + symbol_length, slot_length, symbol_length)

    symbols = np.array(symbols)

    num_non_zero = symbols[symbols != 0].shape[0]

    assert num_non_zero == num_symbols
    assert len(symbols) == len(timestamps)


def test_demodulate_empty_array_raises_exception():
    with pytest.raises(IndexError):
        _ = demodulate([], 16)
