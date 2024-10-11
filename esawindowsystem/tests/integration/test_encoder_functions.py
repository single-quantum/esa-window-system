import numpy as np

from esawindowsystem.core.encoder_functions import (bit_deinterleave, bit_interleave,
                                                    channel_deinterleave, channel_interleave)
from esawindowsystem.ppm_parameters import B_interleaver, M, N_interleaver


def test_channel_interleaving():
    rng = np.random.default_rng()
    input_symbols = rng.integers(0, M, size=100)
    interleaved_symbols = channel_interleave(input_symbols, B_interleaver, N_interleaver)
    deinterleaved_symbols = channel_deinterleave(interleaved_symbols, B_interleaver, N_interleaver)
    deinterleaved_symbols = np.array(deinterleaved_symbols)

    # Number of zeros added due to interleaving
    interleaver_overhead = B_interleaver * N_interleaver * (N_interleaver - 1)

    assert len(interleaved_symbols) - interleaver_overhead == len(input_symbols)
    # The last 2*interleaver_overhead symbols should all be zeros due to interleaving
    assert np.all(deinterleaved_symbols[(len(deinterleaved_symbols) - 2 * interleaver_overhead):] == 0)

    # After deinterleaving, all symbols should be the same
    assert np.all(input_symbols == deinterleaved_symbols[:(len(deinterleaved_symbols) - 2 * interleaver_overhead)])


def test_deinterleaving_longer_message_than_was_interleaved():
    """The goal of this test is to show that when 2 codewords were interleaved, 
    but 3 codewords are deinterleaved, that the symbols end up correctly anyway. """

    M = 8

    rng = np.random.default_rng(777)
    ppm_symbols = rng.integers(0, M, 1000)
    additional_ppm_symbols = rng.integers(0, M, 100)

    interleaved_ppm_symbols = channel_interleave(ppm_symbols, B_interleaver, N_interleaver)

    ppm_symbols_received = np.hstack((interleaved_ppm_symbols, additional_ppm_symbols))
    deinterleaved_ppm_symbols = channel_deinterleave(ppm_symbols_received, B_interleaver, N_interleaver)
    assert np.all(deinterleaved_ppm_symbols[:ppm_symbols.shape[0]] == ppm_symbols)


def test_bit_interleaving():
    rng = np.random.default_rng()

    # The input to the bit interleaver has to be 15120, otherwise the permutation polynomial is not correct
    input_bits = rng.integers(0, 2, 15120)
    interleaved_bits = bit_interleave(input_bits)
    deinterleaved_bits = bit_deinterleave(interleaved_bits)

    assert np.all(deinterleaved_bits == input_bits)
