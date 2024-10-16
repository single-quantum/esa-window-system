from fractions import Fraction

import numpy as np
import pytest

from esawindowsystem.core.encoder_functions import map_PPM_symbols, slot_map, channel_interleave, puncture, slicer, zero_terminate


def test_map_PPM_symbols_3_bits_0():
    input_arr = [0, 0, 0]
    output = map_PPM_symbols(input_arr, 3)
    assert output == [0]


def test_map_PPM_symbols_3_bits_8():
    input_arr = [1, 1, 1]
    output = map_PPM_symbols(input_arr, 3)
    assert output == [7]


def test_map_PPM_symbols_6_bits():
    input_arr = [1, 1, 0, 0, 1, 0]
    output = map_PPM_symbols(input_arr, 3)
    assert list(output) == [6, 2]


def test_map_PPM_symbols_array_too_small():
    input_arr = [1, 1]
    with pytest.raises(ValueError):
        _ = map_PPM_symbols(input_arr, 3)


def test_map_PPM_symbols_array_too_large():
    input_arr = [1, 0, 0, 1]
    with pytest.raises(ValueError):
        _ = map_PPM_symbols(input_arr, 3)


def test_map_PPM_symbols_non_bit_inputs():
    input_arr = [1, 0, 2, 0]
    with pytest.raises(ValueError):
        _ = map_PPM_symbols(input_arr, 3)


def test_slot_map_no_guard_slots():
    input_symbols = [0, 3]
    output_arr = slot_map(input_symbols, 4, insert_guardslots=False)

    assert list(output_arr[0]) == [1, 0, 0, 0]
    assert list(output_arr[1]) == [0, 0, 0, 1]


def test_slot_map_with_guard_slots():
    input_symbols = [0, 1, 2, 3]
    M = 4

    output_arr = slot_map(input_symbols, M, insert_guardslots=True)

    num_guard_slots = M // 4
    assert np.all(output_arr[:, -num_guard_slots:] == 0)


def test_slot_map_PPM_symbol_too_large():
    # This test is a bit artificial, as a PPM symbol that is too high indicate other errors elsewhere,
    # but I want this error to handle it gracefully.
    input_symbols = [8]
    with pytest.raises(ValueError):
        _ = slot_map(input_symbols, 4)


def test_channel_interleaver_one_codeword():
    """Test whether the interleaver produces the correct amount of symbols"""
    input_symbols = np.ones(100, dtype=int)

    M = 8
    m = np.log2(M)

    N_interleaver = 3
    B_interleaver = int((15120 / m) / N_interleaver)
    interleaved_symbols = channel_interleave(input_symbols, B_interleaver, N_interleaver)

    # Assert that there is still the same amount of symbols with the value 1 present in the
    # interleaved codeword, as the interleaver only mixes symbols.
    assert interleaved_symbols[interleaved_symbols == 1].shape[0] == input_symbols.shape[0]
    # Also assert that an additional B*N*(N-1) of zeros were inserted, as specified by
    # the CCSDS protocol
    assert interleaved_symbols.shape[0] == input_symbols.shape[0] + B_interleaver*N_interleaver*(N_interleaver - 1)


def test_slicer_one_slice_one_third_coderate_no_crc():
    input_arr = np.ones(10, dtype=int)
    code_rate = Fraction(1, 3)
    num_termination_bits = 2
    expected_information_block_size = 15120*float(code_rate) - num_termination_bits

    output_arr = slicer(input_arr, code_rate, False, 32, 2)
    # One slice
    assert output_arr.shape[0] == 1
    assert output_arr.shape[1] == expected_information_block_size


def test_slicer_two_slices_one_third_coderate_no_crc():
    code_rate = Fraction(1, 3)
    num_termination_bits = 2
    expected_information_block_size = 15120*float(code_rate) - num_termination_bits

    input_arr = np.ones(int(expected_information_block_size + 1), dtype=int)
    output_arr = slicer(input_arr, code_rate, False, 32, 2)

    assert output_arr.shape[0] == 2
    # Remainder of the slice should be zero filled
    assert np.all(output_arr[1, 1:] == 0)


def test_puncture_one_third_coderate():
    input_arr = np.ones(6, dtype=int)
    code_rate = Fraction(1, 3)

    output_arr = puncture(np.array([input_arr]), code_rate)
    # At 1/3 code rate, no bits should be punctured.
    assert output_arr.shape[1] == 6
    # The data in the output array should not be changed.
    assert np.all(output_arr == 1)


def test_puncture_one_half_coderate():
    input_arr = np.ones(6, dtype=int)
    code_rate = Fraction(1, 2)

    output_arr = puncture(np.array([input_arr]), code_rate)
    # At 1/3 code rate, no bits should be punctured.
    assert output_arr.shape[1] == 4
    # The data in the output array should not be changed.
    assert np.all(output_arr == 1)


def test_puncture_two_third_coderate():
    input_arr = np.ones(6, dtype=int)
    code_rate = Fraction(2, 3)

    output_arr = puncture(np.array([input_arr]), code_rate)
    # At 1/3 code rate, no bits should be punctured.
    assert output_arr.shape[1] == 3
    # The data in the output array should not be changed.
    assert np.all(output_arr == 1)


def test_zero_terminate():
    input_arr = np.ones((2, 12))
    output_arr = zero_terminate(input_arr, 2)
    assert np.all(output_arr[0, -2:] == 0)
    assert np.all(output_arr[1, -2:] == 0)
