# %%
import pickle
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
                                         predict_iteratively,
                                         set_outer_code_gammas)
from core.data_converter import payload_to_bit_sequence
from core.encoder_functions import (bit_deinterleave, bit_interleave,
                                    channel_deinterleave, randomize, slot_map,
                                    unpuncture)
from core.scppm_encoder import encoder, get_csm, puncture
from core.trellis import Trellis
from core.utils import (generate_inner_encoder_edges,
                        generate_outer_code_edges, poisson_noise)

###################
# User parameters #
###################
M: int = 8
ns: float = 2.5      # Average number of photons in the signal slot
nb: float = 0.1    # Average number of photons in the noise slot
code_rate = Fraction(2, 3)
detection_efficiency = 0.85
simulate_lost_symbols = True

greyscale = True
colormap = 'L' if greyscale else '1'

max_num_iterations: int = 5
ber_stop_threshold: float = 1E-3

# bit_stream = np.random.randint(0, 2, 2 * 5038, dtype=int)
image_path = Path.cwd() / 'sample_payloads' / 'herbig-haro-211.jpg'
image = Image.open(image_path).convert(colormap)
bit_stream = payload_to_bit_sequence('image', filepath=image_path)
# pixel_values = np.asarray(image).astype(int).flatten()
# if greyscale:
#     bit_stream = ppm_symbols_to_bit_array(pixel_values, 8)
# else:
#     bit_stream = pixel_values

m = np.log2(M)

information_block_sizes = {
    Fraction(1, 3): 5040,
    Fraction(1, 2): 7560,
    Fraction(2, 3): 10080
}

num_bits_per_slice = information_block_sizes[code_rate]
num_symbols_per_slice = int(num_bits_per_slice * 1 / code_rate / m)

slot_mapped_sequence = encoder(
    bit_stream,
    M, code_rate,
    **{
        'use_randomizer': True,
        'use_inner_encoder': True
    })


ppm_symbols = np.nonzero(slot_mapped_sequence)[1]

CSM = get_csm(M)
ppm_symbols = ppm_symbols.reshape(-1, num_symbols_per_slice + len(CSM))
ppm_symbols = ppm_symbols[:, len(CSM):].flatten()

# Simulate a burst error
# rng = np.random.default_rng()
# len_burst_error = 2        # Number of symbols altered by the burst error
# n0 = rng.integers(0, len(ppm_symbols) - len_burst_error - 1)
# ppm_symbols[n0:n0 + len_burst_error] = rng.integers(0, 1, len_burst_error)

# Deinterleave
N_interleaver = 2
B_interleaver = B_interleaver = int(15120 / m / N_interleaver)
ppm_symbols = channel_deinterleave(ppm_symbols, B_interleaver, N_interleaver)
slot_mapped_sequence = slot_map(ppm_symbols, M, insert_guardslots=True)

decoded_message = []
# This one is used for visualization purposes
num_slices = int((slot_mapped_sequence.shape[0] * m * code_rate) / num_bits_per_slice)
decoded_message_array = np.zeros((max_num_iterations, num_slices, num_bits_per_slice))

with open('sent_bit_sequence_no_csm', 'rb') as f:
    sent_bit_sequence_no_csm = pickle.load(f)

decoded_message, decoded_message_array, bit_error_ratios = predict_iteratively(
    slot_mapped_sequence,
    M,
    code_rate,
    max_num_iterations,
    ns,
    nb,
    **{
        'sent_bit_sequence_no_csm': sent_bit_sequence_no_csm,
        'simulate_lost_symbols': simulate_lost_symbols,
        'detection_efficiency': detection_efficiency
    }
)

information_block_sizes = {
    Fraction(1, 3): 5040,
    Fraction(1, 2): 7560,
    Fraction(2, 3): 10080
}

num_bits = information_block_sizes[code_rate]

# Remove termination bits
# decoded_message = np.array(decoded_message)[:, :-2].flatten()
information_blocks = decoded_message.reshape((-1, num_bits))[:, :-2].flatten()


decoded_message_array = decoded_message_array[:, :, :-2]
decoded_message_array = np.squeeze(decoded_message_array.reshape(max_num_iterations, 1, -1))

total_ber = np.sum([abs(x - y) for x, y in zip(information_blocks, bit_stream)]) / len(bit_stream)
print('total ber', total_ber)

img_shape = image.size
if greyscale:
    pixel_values = map_PPM_symbols(information_blocks[:img_shape[1] * img_shape[0] * 8], 8)
    img_array = pixel_values[:img_shape[1] * img_shape[0]].reshape(img_shape[1], img_shape[0])
else:
    img_array = information_blocks[:img_shape[1] * img_shape[0]].reshape(img_shape[1], img_shape[0])

with open(f'bit_error_ratios_{detection_efficiency*100:.0f}_percent_efficiency', 'wb') as f:
    pickle.dump(bit_error_ratios, f)

plt.figure()
plt.imshow(img_array, cmap='Greys')
plt.show()

plt.figure()
plt.plot(range(max_num_iterations), np.sum(bit_error_ratios, axis=1))
plt.xlabel('Iteration')
plt.ylabel('Bit error ratio')
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
