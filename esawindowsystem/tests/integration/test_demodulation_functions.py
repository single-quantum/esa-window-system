from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from numpy.random import default_rng

from esawindowsystem.core.demodulation_functions import (
    demodulate, determine_CSM_time_shift, find_and_parse_codewords,
    find_csm_times, get_csm_correlation, make_time_series)
from esawindowsystem.core.encoder_functions import get_csm
from esawindowsystem.core.parse_ppm_symbols import parse_ppm_symbols


@pytest.fixture
def pulse_timestamps_no_csm() -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    print(pytest.__version__)
    M = 8
    symbol_length = 5 / 4 * M * 0.1E-6
    slot_length = 0.1E-6
    num_symbols = 10
    CSM = get_csm(M)

    pulse_timestamps: npt.NDArray[np.float64] = np.arange(
        0.5 * slot_length, num_symbols * symbol_length + 0.5 * slot_length, symbol_length)
    ppm_params = {'M': M, 'symbol_length': symbol_length,
                  'slot_length': slot_length, 'num_symbols': num_symbols, 'CSM': CSM}

    return pulse_timestamps, ppm_params


@pytest.fixture
def pulse_timestamps_with_csm() -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
    M = 8
    slot_length = 0.1E-6
    symbol_length = 5 / 4 * M * slot_length
    num_symbols = 5040
    CSM = get_csm(M)
    ppm_params = {'M': M, 'symbol_length': symbol_length,
                  'slot_length': slot_length, 'num_symbols': num_symbols, 'CSM': CSM}

    CSM_timestamps: list[float] = []
    for i, symbol in enumerate(CSM):
        timestamp = i * symbol_length + symbol * slot_length + 0.5 * slot_length
        CSM_timestamps.append(timestamp)

    msg_pulse_timestamps = np.arange(
        start=len(CSM) * symbol_length + 0.5 * slot_length,
        stop=len(CSM) * symbol_length + num_symbols * symbol_length + 0.5 * slot_length,
        step=symbol_length)

    pulse_timestamps = np.hstack((CSM_timestamps, msg_pulse_timestamps))

    return pulse_timestamps, ppm_params


@pytest.fixture
def pulse_timestamps_single_csm() -> tuple[npt.NDArray[np.float64], float, int, float]:
    slot_length = 1E-9
    num_slots_per_symbol = 10
    symbol_length = num_slots_per_symbol * slot_length
    CSM = get_csm(M=8)
    symbols_per_codeword = 100

    time_stamps_1 = np.arange(0, symbols_per_codeword // 2 * symbol_length, symbol_length) + 0.5 * slot_length
    csm_time_stamps = symbols_per_codeword // 2 * symbol_length + \
        np.array([slot_length * CSM[i] + i * symbol_length for i in range(len(CSM))]) + 0.5 * slot_length
    time_stamps_2 = np.arange(50 * symbol_length + len(CSM) * symbol_length, symbols_per_codeword *
                              symbol_length, symbol_length) + 0.5 * slot_length
    time_stamps = np.concatenate((time_stamps_1, csm_time_stamps, time_stamps_2))

    return time_stamps, slot_length, num_slots_per_symbol, symbol_length


def test_parse_ppm_symbols_single_noise_symbol():
    # With a pulse falling in a guard slot, it will be parsed as a 0.
    slot_length = 0.1E-6
    symbol_length = 1E-6
    M = 8
    symbols, _ = parse_ppm_symbols(np.array([symbol_length - slot_length]), 0, 1E-6, slot_length, symbol_length, M)
    assert len(symbols) == 1
    assert symbols[0] == 0


def test_parse_ppm_symbols_single_symbol():
    M = 8
    symbols, _ = parse_ppm_symbols(np.array([0.55E-6]), 0, 1E-6, 0.1E-6, 1E-6, M)
    assert len(symbols) == 1
    # Approx is needed due to floating point representation
    assert symbols[0] == pytest.approx(5)


def test_parse_ppm_symbols_multiple_symbols():
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length
    M = 8
    num_symbols = 100

    timestamps = np.linspace(0, (num_symbols - 1) * symbol_length, num_symbols) + 0.5 * slot_length
    symbols, _ = parse_ppm_symbols(timestamps, 0, timestamps[-1] + symbol_length, slot_length, symbol_length, M)
    symbols = np.round(np.array(symbols)).astype(int)
    print(symbols)

    assert len(symbols) == len(timestamps)
    assert np.all(symbols == 0)


def test_parse_ppm_symbols_multiple_symbols_with_jitter():
    """At this time there is no check for the timing requirement, so no symbols should be dropped. """
    symbol_length = 1E-6
    slot_length = 0.1 * symbol_length
    M = 8
    num_symbols = 10000

    timestamps = np.arange(0, num_symbols * symbol_length, 1E-6) + 0.5 * slot_length + slot_length
    rng = default_rng(777)
    timestamps += rng.normal(0, 0.1 * slot_length, len(timestamps))

    symbols, _ = parse_ppm_symbols(timestamps, 0, timestamps[-1] + symbol_length, slot_length, symbol_length, M)

    symbols = np.array(symbols)

    num_non_zero = symbols[symbols != 0].shape[0]

    assert num_non_zero == num_symbols
    assert len(symbols) == len(timestamps)


def test_demodulate_empty_array_raises_exception():
    with pytest.raises(IndexError):
        _ = demodulate(np.array([]), 16, 1, 1)


def test_demodulate_no_csm_raises_value_error(pulse_timestamps_no_csm: tuple[npt.NDArray[np.float64], dict[str, Any]]):
    pulse_timestamps, ppm_params = pulse_timestamps_no_csm
    M, symbol_length, slot_length, _, _ = ppm_params.values()
    with pytest.raises(ValueError):
        _ = demodulate(pulse_timestamps, M, slot_length, symbol_length)


def test_demodulate_happy_path(pulse_timestamps_with_csm: tuple[npt.NDArray[np.float64], dict[str, Any]]):
    pulse_timestamps, ppm_params = pulse_timestamps_with_csm
    M, symbol_length, slot_length, num_symbols, CSM = ppm_params.values()

    slot_mapped_sequence, _ = demodulate(pulse_timestamps, M, slot_length, symbol_length)
    # The demodulate function demodulates all timestamps, including the CSM.
    # The decoder is responsible for stripping off the CSM.
    assert slot_mapped_sequence.shape[0] == num_symbols + len(CSM)
    assert slot_mapped_sequence.shape[1] == int(5 / 4 * M)


def test_csm_time_shift_happy_path(pulse_timestamps_with_csm: tuple[npt.NDArray[np.float64], dict[str, Any]]):
    pulse_timestamps, ppm_params = pulse_timestamps_with_csm
    M, _, slot_length, _, CSM = ppm_params.values()
    num_slots_per_symbol = int(5 / 4 * M)
    csm_times = np.array([pulse_timestamps[0]])

    rng = np.random.default_rng(777)
    pulse_timestamps += rng.normal(0, 0.1 * slot_length, len(pulse_timestamps))
    csm_time_shift = determine_CSM_time_shift(csm_times, pulse_timestamps, slot_length, CSM, num_slots_per_symbol)
    assert abs(csm_time_shift) == pytest.approx(0.1 * slot_length, abs=0.5 * slot_length)


def test_make_time_series_all_ones():
    slot_length = 1E-9
    time_stamps = np.arange(slot_length, 10 * slot_length, slot_length)
    time_series = make_time_series(time_stamps, slot_length)

    assert len(time_series) == len(time_stamps)
    assert np.all(time_series) == 1


def test_make_time_series():
    slot_length = 1E-9
    symbol_length = 10 * slot_length
    time_stamps = np.arange(0, 10 * symbol_length, symbol_length)

    time_series = make_time_series(time_stamps, slot_length)

    assert time_series[time_series == 1].shape[0] == 10


def test_find_csm_times_one_csm_no_lost_symbols(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    CSM = get_csm(M=8)

    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm
    symbols_per_codeword = len(pulse_timestamps_single_csm)

    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)
    csm_times = find_csm_times(time_stamps, CSM, slot_length,
                               symbols_per_codeword, num_slots_per_symbol, csm_correlation, debug_mode=True)

    assert len(csm_times) == 1


def test_get_csm_correlation_one_csm_no_lost_symbols(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm
    CSM = get_csm(M=8)

    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    assert np.where(csm_correlation > 0.6 * len(CSM))[0].shape[0] == 1
    assert np.where(csm_correlation > 0.6 * len(CSM))[0] == 50 * num_slots_per_symbol
    assert np.max(csm_correlation) == len(CSM) - 1


def test_get_csm_correlation_first_symbol_lost(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm

    time_stamps = np.delete(time_stamps, 50)

    CSM = get_csm(M=8)

    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    assert np.where(csm_correlation > 0.6 * len(CSM))[0].shape[0] == 1
    assert np.where(csm_correlation > 0.6 * len(CSM))[0][0] == 50 * num_slots_per_symbol
    assert np.max(csm_correlation) == len(CSM) - 1


def test_get_csm_correlation_second_symbol_lost(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm

    time_stamps = np.delete(time_stamps, 51)

    CSM = get_csm(M=8)

    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    assert np.where(csm_correlation > 0.6 * len(CSM))[0].shape[0] == 1
    assert np.where(csm_correlation > 0.6 * len(CSM))[0] == 50 * num_slots_per_symbol
    assert np.max(csm_correlation) == len(CSM) - 2


def test_get_csm_correlation_third_symbol_lost(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm

    time_stamps = np.delete(time_stamps, 52)

    CSM = get_csm(M=8)

    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    assert np.where(csm_correlation > 0.6 * len(CSM))[0].shape[0] == 1
    assert np.where(csm_correlation > 0.6 * len(CSM))[0] == 50 * num_slots_per_symbol
    assert np.max(csm_correlation) == len(CSM) - 2


def test_get_csm_correlation_first_symbol_lost_and_noise(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    time_stamps, slot_length, _, symbol_length = pulse_timestamps_single_csm

    time_stamps = np.delete(time_stamps, 52)

    rng = np.random.default_rng(895)
    time_stamps += rng.normal(0, 0.1 * slot_length, size=len(time_stamps))

    CSM = get_csm(M=8)

    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    assert np.where(csm_correlation > 0.6 * len(CSM))[0].shape[0] == 1
    assert np.where(csm_correlation > 0.6 * len(CSM))[0] == 501
    # I should figure out why the maximum is only 13, instead of 16
    assert np.max(csm_correlation) == len(CSM) - 3


def test_find_csm_times_one_csm_first_symbol_lost(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    CSM = get_csm(M=8)

    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm
    symbols_per_codeword = len(time_stamps)

    time_stamps = np.delete(time_stamps, 50)
    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    csm_times = find_csm_times(time_stamps, CSM, slot_length,
                               symbols_per_codeword, num_slots_per_symbol, csm_correlation, debug_mode=True)

    assert len(csm_times) == 1


def test_find_csm_times_and_demodulate_first_symbol_lost(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    CSM = get_csm(M=8)

    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm
    symbols_per_codeword = len(time_stamps)

    time_stamps = np.delete(time_stamps, 50)
    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    csm_times = find_csm_times(time_stamps, CSM, slot_length,
                               symbols_per_codeword, num_slots_per_symbol, csm_correlation, debug_mode=True)

    # Now that the CSM time has been found, demodulate the timestamps. Should give back the CSM
    symbols = find_and_parse_codewords(csm_times, time_stamps, CSM,
                                       symbols_per_codeword, slot_length, symbol_length, M=8)
    assert np.all(symbols[0][:len(CSM)] == CSM)


def test_find_csm_times_and_demodulate_second_symbol_lost(
        pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    CSM = get_csm(M=8)

    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm
    symbols_per_codeword = len(time_stamps)

    time_stamps = np.delete(time_stamps, 51)
    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    csm_times = find_csm_times(time_stamps, CSM, slot_length,
                               symbols_per_codeword, num_slots_per_symbol, csm_correlation, debug_mode=True)

    # Now that the CSM time has been found, demodulate the timestamps. Should give back the CSM
    symbols = find_and_parse_codewords(csm_times, time_stamps, CSM,
                                       symbols_per_codeword, slot_length, symbol_length, M=8)
    assert np.sum(symbols[0][:len(CSM)] == CSM) == len(CSM) - 1


def test_find_csm_times_no_time_shift(pulse_timestamps_single_csm: tuple[npt.NDArray[np.float64], float, int, float]):
    CSM = get_csm(M=8)

    time_stamps, slot_length, num_slots_per_symbol, symbol_length = pulse_timestamps_single_csm
    symbols_per_codeword = len(time_stamps)
    csm_correlation = get_csm_correlation(time_stamps, slot_length, CSM, symbol_length)

    csm_times = find_csm_times(time_stamps, CSM, slot_length,
                               symbols_per_codeword, num_slots_per_symbol, csm_correlation, debug_mode=True)

    assert csm_times[0] == symbols_per_codeword / 2 * symbol_length
