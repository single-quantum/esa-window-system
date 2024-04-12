import numpy as np

from esawindowsystem.core import utils


def test_tobits_space():
    bit_array = utils.tobits(' ')
    assert bit_array == [0, 0, 1, 0, 0, 0, 0, 0]


def test_tobits_a():
    bit_array = utils.tobits('a')
    assert bit_array == [0, 1, 1, 0, 0, 0, 0, 1]


def test_bpsk_encoding_empty_list():
    encoded_array = utils.bpsk_encoding([])
    assert encoded_array.size == 0


def test_bpsk_encoding_list_input():
    encoded_array = utils.bpsk_encoding([1, 2, 3])
    assert np.all(encoded_array == 1)


def test_bpsk_encoding_all_zeros():
    encoded_array = utils.bpsk_encoding(np.zeros(5, dtype=int))
    assert np.all(encoded_array == -1)
