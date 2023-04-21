import copy

import numpy as np
import pytest
from numpy.random import default_rng

from demodulation_functions import check_csm, find_csm_idxs
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


def test_check_csm_indexes_happy_path():
    symbols = CSM
    csm_found = check_csm(symbols)
    assert csm_found


def test_check_csm_indexes_with_lost_symbols():
    symbols = copy.deepcopy(CSM)
    threshold = 0.75
    num_lost_symbols = int((1 - threshold) * len(symbols))

    # Insert wrong symbols on purpose, to make sure they're never part of the CSM
    symbols[:num_lost_symbols] = 99

    csm_found = check_csm(symbols, similarity_threshold=threshold)
    assert csm_found


def test_check_csm_indexes_with_too_many_lost_symbols():
    symbols = copy.deepcopy(CSM)
    threshold = 0.75
    num_lost_symbols = int((1 - threshold) * len(symbols)) + 1

    # Insert wrong symbols on purpose, to make sure they're never part of the CSM
    symbols[:num_lost_symbols] = 99

    csm_found = check_csm(symbols, similarity_threshold=threshold)
    assert not csm_found


def test_check_csm_indexes_all_zeros():
    symbols = np.zeros(len(CSM))
    csm_found = check_csm(symbols)

    assert not csm_found


def test_find_csm_indexes_happy_path():
    symbol_length = num_bins_per_symbol
    slot_length = 1
    time_stamps = [slot_length * CSM[i] + i * symbol_length for i in range(len(CSM))]

    # add some dummy symbols
    for i in range(100):
        time_stamps.append(time_stamps[-1] + (len(time_stamps) + i) * symbol_length)

    time_stamps = np.array(time_stamps) + 0.5 * slot_length

    csm_idxs = find_csm_idxs(time_stamps, CSM, slot_length, symbol_length)
    assert len(csm_idxs) == 1


def test_find_csm_indexes_first_symbol_wrong():
    symbol_length = num_bins_per_symbol
    slot_length = 1
    time_stamps = [slot_length * CSM[i] + i * symbol_length for i in range(len(CSM))]
    time_stamps[0] = slot_length

    # add some dummy symbols
    for i in range(100):
        time_stamps.append(time_stamps[-1] + (len(time_stamps) + i) * symbol_length)

    time_stamps = np.array(time_stamps) + 0.5 * slot_length

    csm_idxs = find_csm_idxs(time_stamps, CSM, slot_length, symbol_length)
    assert len(csm_idxs) == 1


def test_find_csm_indexes_remove_first_symbol():
    symbol_length = num_bins_per_symbol
    slot_length = 1

    # Start range at 1, so that the first symbol is skipped
    time_stamps = [slot_length * CSM[i] + i * symbol_length for i in range(1, len(CSM))]

    # add some dummy symbols
    for i in range(100):
        time_stamps.append(time_stamps[-1] + (len(time_stamps) + i) * symbol_length)

    time_stamps = np.array(time_stamps) + 0.5 * slot_length

    csm_idxs = find_csm_idxs(time_stamps, CSM, slot_length, symbol_length)
    assert len(csm_idxs) == 1


def test_find_csm_indexes_corrupt_several_symbols():
    symbol_length = num_bins_per_symbol
    slot_length = 1

    # Start range at 1, so that the first symbol is skipped
    time_stamps = [slot_length * CSM[i] + i * symbol_length for i in range(0, len(CSM))]

    rng = default_rng(777)

    # add some dummy symbols
    for i in range(100):
        time_stamps.append(time_stamps[-1] + (len(time_stamps) + i) * symbol_length)

    time_stamps[0] = 0 * symbol_length + rng.integers(0, M) * slot_length
    time_stamps[3] = 3 * symbol_length + rng.integers(0, M) * slot_length
    time_stamps[6] = 6 * symbol_length + rng.integers(0, M) * slot_length
    time_stamps = np.array(time_stamps) + 0.5 * slot_length

    csm_idxs = find_csm_idxs(time_stamps, CSM, slot_length, symbol_length)
    assert len(csm_idxs) == 10
