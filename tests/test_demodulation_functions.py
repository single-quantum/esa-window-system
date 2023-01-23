import numpy as np
import pytest
from numpy.random import default_rng

from parse_ppm_symbols import parse_ppm_symbols_new


def test_parse_ppm_symbols_single_noise_symbol():
    # With a pulse falling exactly on the slot edge, it should be parsed as a darkcount.
    symbols = parse_ppm_symbols_new(np.array([0.1E-6]), 0.1E-6, 1E-6)
    assert len(symbols) == 1
    assert symbols[0] == 0


def test_parse_ppm_symbols_single_symbol():
    symbols = parse_ppm_symbols_new(np.array([0.55E-6]), 0.1E-6, 1E-6)
    assert len(symbols) == 1
    # Approx is needed due to floating point representation
    assert symbols[0] == pytest.approx(5)


def test_parse_ppm_symbols_multiple_symbols():
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length

    timestamps = np.linspace(0, 99E-6, 100) + 0.5 * slot_length
    symbols = parse_ppm_symbols_new(timestamps, slot_length, symbol_length)

    assert len(symbols) == len(timestamps)


def test_parse_ppm_symbols_multiple_symbols_with_jitter():
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length

    # The threshold for deciding whether a symbol is a darkcount or not is 3 sigma.
    # Due to this, it is expected that about 0.2% is wrongfully classified as darkcount
    timestamps = np.arange(0, 10000E-6, 1E-6) + 0.5 * slot_length
    rng = default_rng(777)
    timestamps += rng.normal(0, 0.1 * slot_length, len(timestamps))

    symbols = parse_ppm_symbols_new(timestamps, slot_length, symbol_length)

    assert len(symbols) == pytest.approx(0.998 * len(timestamps), rel=1E-3)
