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

ns = 2.0
nb = 0.2

N = 2000

M = 4
code_rate = Fraction(1, 3)
bit_stream = np.random.randint(0, 1, 5038, dtype=int)
# bit_stream[10:10 + N] = np.random.randint(0, 2, N)
# bit_stream[10:20] = 1

slot_mapped_sequence = encoder(bit_stream, M, code_rate)
channel_modulated_sequence = poisson_noise(slot_mapped_sequence[:, :M], ns, nb)
channel_log_likelihoods = pi_ck(channel_modulated_sequence, ns, nb)
# For the first iteration, set the bit_LLRs to 0, update with each iteration
# For each input symbol to the edge, calculate
# np.sum([0.5 * (-1)**PPM_symbol_vector[i] * bit_LLRs[i] for i in range(len(bit_LLRs))])

memory_size = 1
num_output_bits = M // 2
num_input_bits = M // 2

time_steps_inner = slot_mapped_sequence.shape[0]

inner_edges = generate_inner_encoder_edges(M // 2, bpsk_encoding=False)
inner_trellis = Trellis(memory_size, num_output_bits, time_steps_inner, inner_edges, num_input_bits)
inner_trellis.set_edges(inner_edges, zero_terminated=False)


def predict_inner_SISO(trellis, received_sequence, channel_log_likelihoods, time_steps, symbol_bit_LLRs=None):
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

for _ in range(100):
    p_ak_O = predict_inner_SISO(inner_trellis, slot_mapped_sequence,
                                channel_log_likelihoods, time_steps_inner, symbol_bit_LLRs=symbol_bit_LLRs)
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

    u_hat = [0 if llr > 0 else 1 for llr in LLRs_u]
    ber = np.sum([abs(x - y) for x, y in zip(u_hat, bit_stream)]) / len(bit_stream)
    print(f'ber: {ber:.5f} \t min likelihood: {np.min(LLRs_u):.2f} \t max likelihood: {np.max(LLRs_u):.2f}')
    if ber == 0:
        break


print('done')
