import pickle

import numpy as np
import pytest

import esawindowsystem.core.BCJR_decoder_functions as decoder_functions


def test_pi_ck_small_array(benchmark):
    rng = np.random.default_rng(7878)
    arr_shape = (2, 4)
    output_sequence = benchmark(decoder_functions.pi_ck, rng.random(arr_shape), 1, 0.1)

    expected_array = np.array([
        [0.97629779, 0.11333907, 0.29462992, 0.47107697],
        [0.71458503, 0.8380406, 0.53743563, 0.83004967]
    ])

    assert output_sequence.shape == (2, 4)
    np.testing.assert_array_almost_equal(
        expected_array,
        output_sequence,
        decimal=6
    )


def test_pi_ck_big_array(benchmark):
    rng = np.random.default_rng()
    arr_shape = (3760, 16)
    output_sequence = benchmark(decoder_functions.pi_ck, rng.random(arr_shape), 1, 0.1)
    assert output_sequence.shape == arr_shape


@pytest.fixture
def trellis_fixture():
    with open('esawindowsystem/tests/unit/test_file_trellis', 'rb') as f:
        trellis = pickle.load(f)

    with open('esawindowsystem/tests/unit/test_file_symbol_log_likelihoods', 'rb') as f:
        symbol_log_likelihoods = pickle.load(f)

    with open('esawindowsystem/tests/unit/test_file_edge_outputs', 'rb') as f:
        edge_outputs = pickle.load(f)

    return trellis, edge_outputs, symbol_log_likelihoods


def test_outer_code_gammas_cls_based(trellis_fixture, benchmark):
    trellis, _, symbol_log_likelihoods = trellis_fixture

    benchmark(
        decoder_functions.set_outer_code_gammas,
        trellis, symbol_log_likelihoods)
    assert trellis.stages[0].states[0].edges[0].gamma == pytest.approx(-1.99002, rel=1E-5)


def test_outer_code_gammas_arr_based(trellis_fixture, benchmark):
    trellis, edge_outputs, symbol_log_likelihoods = trellis_fixture

    benchmark(
        decoder_functions.get_outer_code_gammas_arr,
        edge_outputs, symbol_log_likelihoods)
    assert True
