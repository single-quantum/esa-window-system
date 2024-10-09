import numpy as np
import pytest
import time

from esawindowsystem.core import utils


def test_tobits_space():
    bit_array = utils.tobits(' ')
    assert bit_array == [0, 0, 1, 0, 0, 0, 0, 0]


def test_tobits_a():
    bit_array = utils.tobits('a')
    assert bit_array == [0, 1, 1, 0, 0, 0, 0, 1]


def test_bpsk_encoding_empty_list(benchmark):
    encoded_array = benchmark(utils.bpsk_encoding, [])
    assert encoded_array.size == 0


def test_bpsk_encoding_list_input(benchmark):
    encoded_array = benchmark(utils.bpsk_encoding, [1, 2, 3])
    assert np.all(encoded_array == 1)


def test_bpsk_encoding_all_zeros_np_array(benchmark):
    encoded_array = benchmark(utils.bpsk_encoding, np.zeros(5, dtype=int))
    assert np.all(encoded_array == -1)


def test_bpsk_encoding_all_zeros_py_list(benchmark):
    encoded_array = benchmark(utils.bpsk_encoding, [0, 0, 0, 0, 0])
    assert np.all(encoded_array == -1)


def test_generate_outer_code_edges_no_bpsk():
    memory_size = 2
    edge_template = utils.generate_outer_code_edges(memory_size, False)
    flattened_edge_template = utils.flatten(edge_template)
    assert len(edge_template) == 2**memory_size
    # The edge template represents the edges belonging to all 4 states (memory size of 2)
    # Because of the convolutional encoder, each state can transition in two ways (input bit 0 or 1)
    # So, the total number of edges should be 8 (4 states x 2 transitions)
    assert len(flattened_edge_template) == 8


def test_generate_outer_code_edges_with_bpsk():
    memory_size = 2
    edge_template = utils.generate_outer_code_edges(memory_size, True)
    flattened_edge_template = utils.flatten(edge_template)
    assert len(edge_template) == 2**memory_size
    # See comment `test_generate_outer_code_edges_no_bpsk`
    assert len(flattened_edge_template) == 8
