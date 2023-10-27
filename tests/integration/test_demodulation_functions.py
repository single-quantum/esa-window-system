import numpy as np
import pytest
from numpy.random import default_rng

from core.demodulation_functions import demodulate, determine_CSM_time_shift
from core.encoder_functions import get_csm
from core.parse_ppm_symbols import parse_ppm_symbols


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
    symbols = np.round(symbols).astype(int)
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
        _ = demodulate([], 16, 1, 1)


@pytest.fixture
def pulse_timestamps_no_csm():
    M = 8
    symbol_length = 5 / 4 * M * 0.1E-6
    slot_length = 0.1E-6
    num_symbols = 10
    CSM = get_csm(M)

    pulse_timestamps = np.arange(0.5 * slot_length, num_symbols * symbol_length + 0.5 * slot_length, symbol_length)
    ppm_params = {'M': M, 'symbol_length': symbol_length,
                  'slot_length': slot_length, 'num_symbols': num_symbols, 'CSM': CSM}

    return pulse_timestamps, ppm_params


@pytest.fixture
def pulse_timestamps_with_csm():
    M = 8
    slot_length = 0.1E-6
    symbol_length = 5 / 4 * M * slot_length
    num_symbols = 5040
    CSM = get_csm(M)
    ppm_params = {'M': M, 'symbol_length': symbol_length,
                  'slot_length': slot_length, 'num_symbols': num_symbols, 'CSM': CSM}

    CSM_timestamps = []
    for i, symbol in enumerate(CSM):
        timestamp = i * symbol_length + symbol * slot_length + 0.5 * slot_length
        CSM_timestamps.append(timestamp)

    msg_pulse_timestamps = np.arange(
        start=len(CSM) * symbol_length + 0.5 * slot_length,
        stop=len(CSM) * symbol_length + num_symbols * symbol_length + 0.5 * slot_length,
        step=symbol_length)

    pulse_timestamps = np.hstack((CSM_timestamps, msg_pulse_timestamps))

    return pulse_timestamps, ppm_params


def test_demodulate_no_csm_raises_value_error(pulse_timestamps_no_csm):
    pulse_timestamps, ppm_params = pulse_timestamps_no_csm
    M, symbol_length, slot_length, _, _ = ppm_params.values()
    with pytest.raises(ValueError):
        _ = demodulate(pulse_timestamps, M, slot_length, symbol_length)


def test_demodulate_happy_path(pulse_timestamps_with_csm):
    pulse_timestamps, ppm_params = pulse_timestamps_with_csm
    M, symbol_length, slot_length, num_symbols, CSM = ppm_params.values()

    slot_mapped_sequence, _ = demodulate(pulse_timestamps, M, slot_length, symbol_length)
    # The demodulate function demodulates all timestamps, including the CSM.
    # The decoder is responsible for stripping off the CSM.
    assert slot_mapped_sequence.shape[0] == num_symbols + len(CSM)
    assert slot_mapped_sequence.shape[1] == int(5 / 4 * M)


def test_csm_time_shift_happy_path(pulse_timestamps_with_csm):
    pulse_timestamps, ppm_params = pulse_timestamps_with_csm
    M, symbol_length, slot_length, num_symbols, CSM = ppm_params.values()
    num_slots_per_symbol = int(5/4*M)
    csm_times = np.array([pulse_timestamps[0]])

    rng = np.random.default_rng(777)
    pulse_timestamps += rng.normal(0, 0.1*slot_length, len(pulse_timestamps))
    csm_time_shift = determine_CSM_time_shift(csm_times, pulse_timestamps, slot_length, CSM, num_slots_per_symbol)
    assert csm_time_shift == 0
