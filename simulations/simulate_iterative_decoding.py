# %%
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


def predict_inner_SISO(trellis, channel_log_likelihoods, time_steps, m, symbol_bit_LLRs=None):
    if symbol_bit_LLRs is None:
        symbol_bit_LLRs = np.zeros((time_steps, m))

    # Calculate alphas, betas, gammas and LLRs
    calculate_gamma_inner_SISO(trellis, symbol_bit_LLRs, channel_log_likelihoods)
    gamma_primes = calculate_gamma_primes(trellis)

    calculate_alpha_inner_SISO(trellis, gamma_primes)
    calculate_beta_inner_SISO(trellis, gamma_primes)
    LLRs = calculate_inner_SISO_LLRs(trellis, symbol_bit_LLRs)

    return LLRs


###################
# User parameters #
###################
M: int = 8
ns: float = 2      # Average number of photons in the signal slot
nb: float = 0.3    # Average number of photons in the noise slot
code_rate = Fraction(1, 3)

max_num_iterations: int = 20
ber_stop_threshold: float = 1E-3

N = 100
bit_stream = np.zeros(5038, dtype=int)
# bit_stream[10:10 + N] = np.random.randint(0, 2, N)
# bit_stream = np.random.randint(0, 2, 2 * 5038, dtype=int)
image_path = Path.cwd() / 'sample_payloads' / 'pillars-of-creation-tiny.png'
image = Image.open(image_path).convert('1')
bit_stream = np.asarray(image).astype(int).flatten()

# Initialize outer trellis edges
memory_size_outer = 2
num_output_bits_outer = 3
num_input_bits_outer = 1
outer_edges = generate_outer_code_edges(memory_size_outer, bpsk_encoding=False)

# Initialize inner trellis edges
m = int(np.log2(M))
memory_size = 1
num_output_bits = m
num_input_bits = m
inner_edges = generate_inner_encoder_edges(m, bpsk_encoding=False)

num_bits_per_slice = 5040
num_symbols_per_slice = int(num_bits_per_slice * 3 / m)


slot_mapped_sequence = encoder(bit_stream, M, code_rate)
num_slices = int((slot_mapped_sequence.shape[0] * m // 3) / num_bits_per_slice)

decoded_message = []

channel_modulated_sequence = poisson_noise(
    slot_mapped_sequence[:, :M], ns, nb)

for i in range(num_slices):
    print(f'Decoding slice {i+1}/{num_slices}')
    # Generate a vector with a poisson distributed number of photons per slot
    # Calculate the corresponding log likelihood
    channel_log_likelihoods = pi_ck(
        channel_modulated_sequence[i * num_symbols_per_slice:(i + 1) * num_symbols_per_slice], ns, nb)

    time_steps_inner = num_symbols_per_slice

    inner_trellis = Trellis(memory_size, num_output_bits, time_steps_inner, inner_edges, num_input_bits)
    inner_trellis.set_edges(inner_edges, zero_terminated=False)

    symbol_bit_LLRs = None

    for _ in range(max_num_iterations):
        p_ak_O = predict_inner_SISO(inner_trellis, channel_log_likelihoods,
                                    time_steps_inner, m, symbol_bit_LLRs=symbol_bit_LLRs)
        p_xk_I = bit_deinterleave(p_ak_O.flatten(), dtype=float)
        # p_xk_I = p_ak_O.flatten()

        time_steps_outer_trellis = len(p_xk_I) // 3

        outer_trellis = Trellis(memory_size_outer, num_output_bits_outer,
                                time_steps_outer_trellis, outer_edges, num_input_bits_outer)
        outer_trellis.set_edges(outer_edges)
        set_outer_code_gammas(outer_trellis, p_xk_I)
        calculate_alphas(outer_trellis)
        calculate_betas(outer_trellis)
        p_xk_O, LLRs_u = calculate_outer_SISO_LLRs(outer_trellis, p_xk_I)
        p_ak_I = bit_interleave(p_xk_O.flatten(), dtype=float)

        symbol_bit_LLRs = deepcopy(p_ak_I.reshape(-1, m))

        u_hat = [0 if llr > 0 else 1 for llr in LLRs_u]
        ber = np.sum(
            [abs(x - y) for x, y in zip(
                u_hat, bit_stream[i * num_bits_per_slice - 2 * i:(i + 1) * num_bits_per_slice - 2 * (i + 1)]
            )]
        ) / len(bit_stream)
        print(
            f'iteration = {i} ber: {ber:.5f} \t min likelihood: {np.min(LLRs_u):.2f} \t max likelihood: {np.max(LLRs_u):.2f}')
        if ber < ber_stop_threshold:
            break

    decoded_message.append(u_hat)

decoded_message = np.array(decoded_message)[:, :-2].flatten()
total_ber = np.sum([abs(x - y) for x, y in zip(decoded_message, bit_stream)])
print('total ber', total_ber)

img_shape = image.size
plt.imshow(decoded_message[:img_shape[0] * img_shape[1]].reshape(img_shape[0], img_shape[1]))
plt.show()

print('done')
