# %%
from copy import deepcopy
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from core.BCJR_decoder_functions import (calculate_alphas, calculate_betas,
                                         calculate_outer_SISO_LLRs,
                                         map_PPM_symbols, pi_ck,
                                         ppm_symbols_to_bit_array,
                                         predict_inner_SISO,
                                         set_outer_code_gammas)
from core.encoder_functions import (bit_deinterleave, bit_interleave,
                                    channel_deinterleave, randomize, slot_map,
                                    unpuncture)
from core.scppm_encoder import encoder, puncture
from core.trellis import Trellis
from core.utils import (generate_inner_encoder_edges,
                        generate_outer_code_edges, poisson_noise)

###################
# User parameters #
###################
M: int = 32
ns: float = 3.5      # Average number of photons in the signal slot
nb: float = 0.01    # Average number of photons in the noise slot
code_rate = Fraction(2, 3)
detection_efficiency = 0.8

greyscale = True
colormap = 'L' if greyscale else '1'

max_num_iterations: int = 5
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


ppm_symbols = np.nonzero(slot_mapped_sequence)[1]

# Simulate a burst error
rng = np.random.default_rng()
len_burst_error = 2        # Number of symbols altered by the burst error
n0 = rng.integers(0, len(ppm_symbols) - len_burst_error - 1)
ppm_symbols[n0:n0 + len_burst_error] = rng.integers(0, 1, len_burst_error)

# Deinterleave
N_interleaver = 2
B_interleaver = B_interleaver = int(15120 / m / N_interleaver)
ppm_symbols = channel_deinterleave(ppm_symbols, B_interleaver, N_interleaver)
slot_mapped_sequence = slot_map(ppm_symbols, M, insert_guardslots=False)

decoded_message = []
# This one is used for visualization purposes
num_slices = int((slot_mapped_sequence.shape[0] * m * code_rate) / num_bits_per_slice)
decoded_message_array = np.zeros((max_num_iterations, num_slices, num_bits_per_slice))

channel_likelihoods = poisson_noise(
    slot_mapped_sequence[:, :M], ns, nb, simulate_lost_symbols=True, detection_efficiency=detection_efficiency)

for i in range(num_slices):
    print(f'Decoding slice {i+1}/{num_slices}')
    # Generate a vector with a poisson distributed number of photons per slot
    # Calculate the corresponding log likelihood
    channel_log_likelihoods = pi_ck(
        channel_likelihoods[i * num_symbols_per_slice:(i + 1) * num_symbols_per_slice], ns, nb)

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
            f"iteration = {iteration+1} ber: {ber:.5f} \t min likelihood: " +
            f"{np.min(LLRs_u):.2f} \t max likelihood: {np.max(LLRs_u):.2f}")

        decoded_message_array[iteration, i, :] = u_hat

        # if ber < ber_stop_threshold:
        #     break

    decoded_message.append(u_hat)

# Remove termination bits
decoded_message = np.array(decoded_message)[:, :-2].flatten()
decoded_message_array = decoded_message_array[:, :, :-2]
decoded_message_array = np.squeeze(decoded_message_array.reshape(max_num_iterations, 1, -1))

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

for i in range(max_num_iterations):
    if greyscale:
        pixel_values = map_PPM_symbols(decoded_message_array[i, :img_shape[1] * img_shape[0] * 8], 8)
        img_array = pixel_values[:img_shape[1] * img_shape[0]].reshape(img_shape[1], img_shape[0])
    else:
        img_array = decoded_message_array[i, :img_shape[1] * img_shape[0]].reshape(img_shape[1], img_shape[0])
    plt.figure()
    plt.imshow(img_array, cmap='Greys')
    plt.title(f'Iteration = {i+1}')

plt.show()
print('done')
