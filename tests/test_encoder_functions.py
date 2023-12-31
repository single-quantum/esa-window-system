import numpy as np
import pytest

from core.encoder_functions import map_PPM_symbols, slot_map


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


def test_map_PP_symbols_array_too_small():
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
