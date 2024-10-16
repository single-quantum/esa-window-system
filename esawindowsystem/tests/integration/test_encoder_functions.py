from fractions import Fraction
from math import ceil

import numpy as np

from esawindowsystem.core.encoder_functions import (bit_deinterleave, bit_interleave,
                                                    channel_deinterleave, channel_interleave)
from esawindowsystem.core.scppm_encoder import encoder
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


def test_encoder_num_output_symbols():
    M = 8

    num_input_bits = 76000
    input_bit_array = np.ones(num_input_bits, dtype=int)
    code_rate = Fraction(2, 3)

    m = np.log2(M)
    N = 2           # Number of parallel shift registers in channel interleaver
    B = 15120/m/N   # Base length of the shift registers

    # This calculation can be done by hand, following the protocol step by step.
    information_block_size = 15120*code_rate                                    # 10080 for 2/3 code rate
    information_block_size_no_termination_bits = information_block_size - 2

    num_information_blocks = ceil(num_input_bits/information_block_size_no_termination_bits)    # 8 information blocks
    num_bits_into_convolutional_encoder = num_information_blocks*information_block_size

    # For each bit going in, three bits come out and afterwards the codewords are punctured
    num_bits_after_convolution = num_bits_into_convolutional_encoder*3

    # For 2/3 code rate, half the bits are punctured.
    num_bits_after_puncturing = num_bits_after_convolution/(3*code_rate)
    message_ppm_symbols = num_bits_after_puncturing/m

    # As specified by the protocol. This is because the channel interleaver needs
    # extra steps to make sure the last symbol comes out.
    num_ppm_symbols_after_interleaving = message_ppm_symbols + B*N*(N-1)
    num_ppm_codewords = num_ppm_symbols_after_interleaving / (15120 / m)

    # For 8 ppm, the CSM is 16 symbols
    num_ppm_symbols_with_csm = num_ppm_symbols_after_interleaving + num_ppm_codewords*16

    expected_number_of_symbols = num_ppm_symbols_with_csm

    slot_mapped_sequence, _, information_blocks = encoder(
        input_bit_array, M, code_rate,
        **{'B_interleaver': B,
           'N_interleaver': N}
    )

    assert information_blocks.shape[0] == num_information_blocks
    assert information_blocks.shape[1] == information_block_size

    assert slot_mapped_sequence.shape[0] == expected_number_of_symbols
    # Guard slots should be included at this point
    assert slot_mapped_sequence.shape[1] == int(5/4*M)
