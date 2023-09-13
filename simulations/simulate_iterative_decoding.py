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
                                         calculate_outer_SISO_LLRs,
                                         map_PPM_symbols, pi_ck,
                                         ppm_symbols_to_bit_array,
                                         set_outer_code_gammas)
from core.encoder_functions import (bit_deinterleave, bit_interleave,
                                    randomize, unpuncture)
from core.scppm_encoder import encoder, puncture
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
M: int = 16
ns: float = 2      # Average number of photons in the signal slot
nb: float = 0.05    # Average number of photons in the noise slot
code_rate = Fraction(2, 3)

greyscale = False
colormap = 'L' if greyscale else '1'

max_num_iterations: int = 20
ber_stop_threshold: float = 1E-3

# bit_stream = np.random.randint(0, 2, 2 * 5038, dtype=int)
image_path = Path.cwd() / 'sample_payloads' / 'JWST_2022-07-27_Jupiter_tiny.png'
image = Image.open(image_path).convert(colormap)
pixel_values = np.asarray(image).astype(int).flatten()
if greyscale:
    bit_stream = ppm_symbols_to_bit_array(pixel_values, 8)
else:
    bit_stream = pixel_values

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

information_block_sizes = {
    Fraction(1, 3): 5040,
    Fraction(1, 2): 7560,
    Fraction(2, 3): 10080
}

num_bits_per_slice = information_block_sizes[code_rate]
num_symbols_per_slice = int(num_bits_per_slice * 1 / code_rate / m)

slot_mapped_sequence = encoder(bit_stream, M, code_rate)
num_slices = int((slot_mapped_sequence.shape[0] * m * code_rate) / num_bits_per_slice)

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
    u_hat = []

    for iteration in range(max_num_iterations):
        p_ak_O = predict_inner_SISO(inner_trellis, channel_log_likelihoods,
                                    time_steps_inner, m, symbol_bit_LLRs=symbol_bit_LLRs)
        p_xk_I = bit_deinterleave(p_ak_O.flatten(), dtype=float)
        # p_xk_I = p_ak_O.flatten()

        p_xk_I = unpuncture(p_xk_I, code_rate, dtype=float)
        time_steps_outer_trellis = int(len(p_xk_I) / 3)

        outer_trellis = Trellis(memory_size_outer, num_output_bits_outer,
                                time_steps_outer_trellis, outer_edges, num_input_bits_outer)
        outer_trellis.set_edges(outer_edges)
        set_outer_code_gammas(outer_trellis, p_xk_I)
        calculate_alphas(outer_trellis)
        calculate_betas(outer_trellis)
        p_xk_O, LLRs_u = calculate_outer_SISO_LLRs(outer_trellis, p_xk_I)
        p_xk_O = puncture(np.array([p_xk_O.flatten()]), code_rate, dtype=float)
        p_ak_I = bit_interleave(p_xk_O.flatten(), dtype=float)

        symbol_bit_LLRs = deepcopy(p_ak_I.reshape(-1, m))

        u_hat = [0 if llr > 0 else 1 for llr in LLRs_u]
        # Derandomize
        u_hat = randomize(np.array(u_hat, dtype=int))
        ber = np.sum(
            [abs(x - y) for x, y in zip(
                u_hat, bit_stream[i * num_bits_per_slice - 2 * i:(i + 1) * num_bits_per_slice - 2 * (i + 1)]
            )]
        ) / len(bit_stream)
        print(
            f"iteration = {iteration} ber: {ber:.5f} \t min likelihood: " +
            f"{np.min(LLRs_u):.2f} \t max likelihood: {np.max(LLRs_u):.2f}")
        if ber < ber_stop_threshold:
            break

    decoded_message.append(u_hat)

decoded_message = np.array(decoded_message)[:, :-2].flatten()
total_ber = np.sum([abs(x - y) for x, y in zip(decoded_message, bit_stream)]) / len(bit_stream)
print('total ber', total_ber)

img_shape = image.size
if greyscale:
    pixel_values = map_PPM_symbols(decoded_message[:img_shape[1] * img_shape[0] * 8], 8)
    img_array = pixel_values[:img_shape[1] * img_shape[0]].reshape(img_shape[1], img_shape[0])
else:
    img_array = decoded_message[:img_shape[1] * img_shape[0]].reshape(img_shape[1], img_shape[0])
plt.imshow(img_array, cmap='Greys')
plt.show()

print('done')
