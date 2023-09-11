# %%
from copy import deepcopy
from fractions import Fraction

import numpy as np

from core.BCJR_decoder_functions import (calculate_alpha_inner_SISO,
                                         calculate_alphas,
                                         calculate_beta_inner_SISO,
                                         calculate_betas,
                                         calculate_gamma_inner_SISO,
                                         calculate_gamma_primes,
                                         calculate_inner_SISO_LLRs,
                                         calculate_outer_SISO_LLRs, pi_ck,
                                         set_outer_code_gammas)
from core.encoder_functions import bit_deinterleave, bit_interleave
from core.scppm_encoder import encoder
from core.trellis import Trellis
from core.utils import (generate_inner_encoder_edges,
                        generate_outer_code_edges, poisson_noise)

ns = 1.9
nb = 0.2

M = 4
code_rate = Fraction(2, 3)
bit_stream = np.zeros(7560, dtype=int)


slot_mapped_sequence = encoder(bit_stream, M, code_rate)
channel_modulated_sequence = poisson_noise(slot_mapped_sequence[:, :M], ns, nb)
channel_log_likelihoods = pi_ck(channel_modulated_sequence, ns, nb)
# For the first iteration, set the bit_LLRs to 0, update with each iteration
# For each input symbol to the edge, calculate
# np.sum([0.5 * (-1)**PPM_symbol_vector[i] * bit_LLRs[i] for i in range(len(bit_LLRs))])

memory_size = 1
num_output_bits = M // 2
num_input_bits = M // 2

time_steps = int(slot_mapped_sequence.shape[0] / (np.log2(M)))

inner_edges = generate_inner_encoder_edges(M // 2, bpsk_encoding=False)
inner_trellis = Trellis(memory_size, num_output_bits, time_steps, inner_edges, num_input_bits)
inner_trellis.set_edges(inner_edges, zero_terminated=False)


def predict_inner_SISO(trellis, received_sequence, channel_log_likelihoods, symbol_bit_LLRs=None):
    time_steps = len(received_sequence)

    if symbol_bit_LLRs is None:
        symbol_bit_LLRs = np.zeros((time_steps, 2))

    # Calculate alphas, betas, gammas and LLRs
    calculate_gamma_inner_SISO(trellis, symbol_bit_LLRs, channel_log_likelihoods)
    gamma_primes = calculate_gamma_primes(trellis)

    calculate_alpha_inner_SISO(trellis, gamma_primes)
    calculate_beta_inner_SISO(trellis, gamma_primes)
    LLRs = calculate_inner_SISO_LLRs(trellis, symbol_bit_LLRs)

    return LLRs


symbol_bit_LLRs = None

llrs = []

memory_size_outer = 2
num_output_bits_outer = 3
num_input_bits_outer = 1
outer_edges = generate_outer_code_edges(memory_size_outer, bpsk_encoding=False)

for _ in range(10):
    p_ak_O = predict_inner_SISO(inner_trellis, slot_mapped_sequence,
                                channel_log_likelihoods, symbol_bit_LLRs=symbol_bit_LLRs)
    p_xk_I = bit_deinterleave(p_ak_O.flatten(), dtype=float)
    # p_xk_I = p_ak_O.flatten()

    time_steps_outer_trellis = len(p_xk_I) // 3

    outer_trellis = Trellis(memory_size_outer, num_output_bits_outer,
                            time_steps_outer_trellis, outer_edges, num_input_bits_outer)
    outer_trellis.set_edges(outer_edges)
    set_outer_code_gammas(outer_trellis, p_xk_I)
    calculate_alphas(outer_trellis)
    calculate_betas(outer_trellis)
    LLRs, LLRs_u = calculate_outer_SISO_LLRs(outer_trellis, p_xk_I)
    LLRs = bit_interleave(LLRs.flatten(), dtype=float)
    llrs.append(LLRs[100])
    symbol_bit_LLRs = deepcopy(LLRs.reshape(-1, 2))

print('done')
