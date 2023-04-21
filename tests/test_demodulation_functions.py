import copy

import numpy as np
import pytest
from numpy.random import default_rng

from demodulation_functions import find_csm_idxs
from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import CSM, M, num_bins_per_symbol


def test_parse_ppm_symbols_single_noise_symbol():
    # With a pulse falling exactly on the slot edge, it should be parsed as a darkcount.
    symbols = parse_ppm_symbols(np.array([0.1E-6]), 0.1E-6, 1E-6)
    assert len(symbols) == 1
    assert symbols[0] == 0


def test_parse_ppm_symbols_single_symbol():
    symbols = parse_ppm_symbols(np.array([0.55E-6]), 0.1E-6, 1E-6)
    assert len(symbols) == 1
    # Approx is needed due to floating point representation
    assert symbols[0] == pytest.approx(5)


def test_parse_ppm_symbols_multiple_symbols():
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length

    timestamps = np.linspace(0, 99E-6, 100) + 0.5 * slot_length
    symbols = parse_ppm_symbols(timestamps, slot_length, symbol_length)
    symbols = np.round(symbols).astype(int)

    assert len(symbols) == len(timestamps)
    assert np.all(symbols == 0)


def test_parse_ppm_symbols_multiple_symbols_with_jitter():
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length

    # The threshold for deciding whether a symbol is a darkcount or not is 3 sigma.
    # Due to this, it is expected that about 0.2% is wrongfully classified as darkcount
    timestamps = np.arange(0, 10000E-6, 1E-6) + 0.5 * slot_length + slot_length
    rng = default_rng(777)
    timestamps += rng.normal(0, 0.1 * slot_length, len(timestamps))

    symbols = parse_ppm_symbols(timestamps, slot_length, symbol_length)

    symbols = np.array(symbols)
    print(np.round(symbols[:10]))
    num_non_zero = symbols[symbols != 0].shape[0]

    assert num_non_zero == pytest.approx(0.998 * len(symbols), rel=1E-3)
    assert len(symbols) == len(timestamps)
