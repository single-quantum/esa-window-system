import pytest
import numpy as np
import esawindowsystem.core.BCJR_decoder_functions as decoder_functions


def test_pi_ck_small_array(benchmark):
    rng = np.random.default_rng(7878)
    arr_shape = (2, 4)
    output_sequence = benchmark(decoder_functions.pi_ck, rng.random(arr_shape), 1, 0.1)

    expected_array = np.array([
        [0.97629779, 0.11333907, 0.29462992, 0.47107697],
        [0.71458503, 0.8380406,  0.53743563, 0.83004967]
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
