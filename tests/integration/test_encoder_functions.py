import numpy as np

from encoder_functions import channel_deinterleave, channel_interleave, bit_interleave, bit_deinterleave

from ppm_parameters import M, B_interleaver, N_interleaver

def test_channel_interleaving():
    rng = np.random.default_rng()
    input_symbols = rng.integers(0, M, size=100)
    interleaved_symbols = channel_interleave(input_symbols, B_interleaver, N_interleaver)
    deinterleaved_symbols = channel_deinterleave(interleaved_symbols, B_interleaver, N_interleaver)
    deinterleaved_symbols = np.array(deinterleaved_symbols)
    
    # Number of zeros added due to interleaving
    interleaver_overhead = B_interleaver*N_interleaver*(N_interleaver-1)

    assert len(interleaved_symbols) - interleaver_overhead == len(input_symbols) 
    # The last 2*interleaver_overhead symbols should all be zeros due to interleaving
    assert np.all(deinterleaved_symbols[(len(deinterleaved_symbols)-2*interleaver_overhead):] == 0)
    
    # After deinterleaving, all symbols should be the same
    assert np.all(input_symbols == deinterleaved_symbols[:(len(deinterleaved_symbols)-2*interleaver_overhead)])

def test_bit_interleaving():
    rng = np.random.default_rng()

    # The input to the bit interleaver has to be 15120, otherwise the permutation polynomial is not correct
    input_bits = rng.integers(0, 2, 15120)
    interleaved_bits = bit_interleave(input_bits)
    deinterleaved_bits = bit_deinterleave(interleaved_bits)

    assert np.all(deinterleaved_bits == input_bits)