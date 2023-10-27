# %%
import itertools
import math
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from PIL import Image

from core.BCJR_decoder_functions import (calculate_alphas, calculate_betas,
                                         calculate_gammas, calculate_LLRs,
                                         ppm_symbols_to_bit_array)
from core.encoder_functions import (bit_deinterleave, bit_interleave,
                                    channel_deinterleave, channel_interleave,
                                    convolve, get_csm, map_PPM_symbols, slicer,
                                    zero_terminate)
from core.parse_ppm_symbols import rolling_window
from core.trellis import Edge, Trellis
from core.utils import AWGN, bpsk, bpsk_encoding, generate_outer_code_edges
from simulations.viterbi import viterbi

GREYSCALE = True
PAYLOAD_TYPE = 'image'
LOG_BCRJ = True
SIMULATE_BURST_ERRORS = False
SIMULATE_LOST_SYMBOLS = False

M = 16      # each m = 4 bits are mapped from 0 to M = 16
m = int(np.log(M) / np.log(2))
num_output_bits = 3    # number of output bits from the convolutional encoder
num_input_bits = 1
memory_size = 2   # Memory size of the convolutional encoder

if GREYSCALE:
    IMG_MODE = "L"
else:
    IMG_MODE = "1"

match PAYLOAD_TYPE:
    case 'image':
        file = "sample_payloads/JWST_2022-07-27_Jupiter_tiny.png"
        img = Image.open(file)
        img = img.convert(IMG_MODE)
        img_array = np.asarray(img).astype(int)
        original_shape = img_array.shape
        # In the case of greyscale, each pixel has a value from 0 to 255.
        # This would be the same as saying that each pixel is a symbol, which should be mapped to an 8 bit sequence.
        if GREYSCALE:
            sent_message = ppm_symbols_to_bit_array(img_array.flatten(), 8).astype(int)
        else:
            sent_message = img_array.flatten().astype(int)
    case 'string':
        sent_message = [1, 1, 0, 1]
        # sent_message = tobits('Hello')
        # sent_message = np.random.randint(0, 2, size=10)
        sent_message = np.append(sent_message, [0, 0])
        print('sent message', sent_message)


def gamma_poisson(ns, nb, num_photons, noise=False):
    if noise:
        lmbda = nb
        return np.exp(-lmbda) * np.sum([lmbda**k / math.factorial(k) for k in range(num_photons)])
    else:
        lmbda = ns
        return np.exp(-lmbda) * np.sum([lmbda**k / math.factorial(k) for k in range(num_photons)])


num_output_bits = 3
num_input_bits = 1
memory_size = 2

edges = generate_outer_code_edges(memory_size, bpsk_encoding=False)


information_blocks = slicer(sent_message, code_rate=Fraction(1, 3))
information_blocks = zero_terminate(information_blocks)

print('Number of slices: ', information_blocks.shape[0])


# The convolutional encoder is a 1/3 code rate encoder.
convoluted_bit_sequence = np.zeros((information_blocks.shape[0], information_blocks.shape[1] * 3), dtype=int)

for i, row in enumerate(information_blocks):
    convoluted_bit_sequence[i], _ = convolve(row)
    convoluted_bit_sequence[i] = bit_interleave(convoluted_bit_sequence[i])


convoluted_bit_sequence = convoluted_bit_sequence.flatten()
B_interleaver = 378
N_interleaver = 10

time_steps = len(convoluted_bit_sequence) // num_output_bits


S = len(convoluted_bit_sequence) // m

ppm_mapped_message = map_PPM_symbols(convoluted_bit_sequence, m)
ppm_mapped_message = channel_interleave(ppm_mapped_message, B_interleaver, N_interleaver)

# Insert CSMs
len_codeword = 15120 // m
num_codewords = ppm_mapped_message.shape[0] // len_codeword
CSM = get_csm(M=M)

ppm_mapped_message_with_csm = np.zeros(len(ppm_mapped_message) + len(CSM) * num_codewords, dtype=int)
for i in range(num_codewords):
    prepended_codeword = np.hstack((CSM, ppm_mapped_message[i * len_codeword:(i + 1) * len_codeword]))
    ppm_mapped_message_with_csm[i * len(prepended_codeword):(i + 1) * len(prepended_codeword)] = prepended_codeword

ppm_mapped_message = ppm_mapped_message_with_csm

rng = default_rng()
if SIMULATE_LOST_SYMBOLS:
    # Simulate some lost symbols
    symbol_noise_factor = 0.00
    num_noise_symbols = int(len(ppm_mapped_message) * symbol_noise_factor)
    symbol_idxs = rng.integers(0, num_noise_symbols, num_noise_symbols)
    noise_symbols = rng.integers(0, M + 1, num_noise_symbols)

    for i, idx in enumerate(symbol_idxs):
        ppm_mapped_message[idx] = noise_symbols[i]

if SIMULATE_BURST_ERRORS:
    # Simulate burst error
    num_burst_errors = 0
    burst_error_length = 20
    burst_error_idxs = rng.integers(0, num_noise_symbols, num_burst_errors)
    for burst_error_idx in burst_error_idxs:
        ppm_mapped_message[burst_error_idx:burst_error_idx + burst_error_length] = np.zeros(burst_error_length)

Es = 4  # Signal power
N0 = 1  # Noise power

SNR = []
BER_BCJR = []
BER_viterbi = []

N = np.arange(5, 5.4, 0.2)
E = np.ones(N.shape[0])

SNR_e = 10 * np.log10(E[-1] / N[-1])
sigma_e = np.sqrt(1 / (2 * 3 * E[-1] / N[-1]))

# The following three lines are for visualization purposes only
bpsk_sent_message = bpsk_encoding(sent_message)
bpsk_sent_message = AWGN(bpsk_sent_message, sigma_e)
# noisy_sent_message = np.array([0 if i < 0 else 1 for i in bpsk_sent_message])

noisy_sent_message = np.zeros_like(bpsk_sent_message)
noisy_sent_message[np.where(bpsk_sent_message < 0)] = 0
noisy_sent_message[np.where(bpsk_sent_message >= 0)] = 1

# Simulate decoding
# Set up the trellis
num_states = 2**memory_size
tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
tr.set_edges(edges)

# Remove CSMs
where_csm = np.where(np.all(rolling_window(ppm_mapped_message, len(CSM)) == CSM, axis=1))[0]
idxs = [np.arange(csm_idx, csm_idx + len(CSM), 1) for csm_idx in where_csm]
idxs_to_be_removed = np.hstack(idxs)

ppm_mapped_message = np.delete(ppm_mapped_message, idxs_to_be_removed)

# Deinterleave
ppm_mapped_message = channel_deinterleave(ppm_mapped_message, B_interleaver, N_interleaver)
# Due to interleaving, two times B*N*(N-1) zeros are added to the ppm message,
# one time for interleaving, and one time for deinterleaving
num_zeros_interleaver = (2 * B_interleaver * N_interleaver * (N_interleaver - 1))
convoluted_bit_sequence = ppm_symbols_to_bit_array(
    ppm_mapped_message[:(len(ppm_mapped_message) - num_zeros_interleaver)], m)

received_sequence_interleaved = convoluted_bit_sequence.reshape((-1, 15120))
received_sequence = np.zeros_like(received_sequence_interleaved)

for i, row in enumerate(received_sequence_interleaved):
    received_sequence[i] = bit_deinterleave(row)

deinterleaved_received_sequence = received_sequence.flatten()
# deinterleaved_received_sequence = np.hstack((deinterleaved_received_sequence, [0, 0]))

for Es, N0 in list(zip(E, N)):
    SNR.append(10 * np.log10(Es / N0))
    sigma = np.sqrt(1 / (2 * 3 * Es / N0))

    encoded_sequence = bpsk_encoding(deinterleaved_received_sequence.astype(float))
    encoded_sequence = AWGN(encoded_sequence, sigma)

    alpha = np.zeros((num_states, time_steps + 1))
    beta = np.zeros((num_states, time_steps + 1))

    predicted_msg_viterbi = viterbi(num_output_bits, encoded_sequence, tr)

    # Calculate alphas, betas, gammas and LLRs
    gammas = calculate_gammas(tr, encoded_sequence, num_output_bits, Es, N0, log_bcjr=LOG_BCRJ, verbose=True)
    alpha = calculate_alphas(tr, alpha, log_bcjr=LOG_BCRJ, verbose=True)
    beta = calculate_betas(tr, beta, log_bcjr=LOG_BCRJ, verbose=True)
    LLR = calculate_LLRs(tr, alpha, beta, log_bcjr=LOG_BCRJ, verbose=True)

    predicted_msg = np.array([1 if l >= 0 else 0 for l in LLR])

    # By reshaping, each row represents an information block
    predicted_msg = predicted_msg.reshape((-1, 5040))
    # Remove termination bits
    predicted_msg = predicted_msg[:, 0:-2].flatten()

    predicted_msg_viterbi = predicted_msg_viterbi.reshape((-1, 5040))
    predicted_msg_viterbi = predicted_msg_viterbi[:, 0:-2].flatten()

    match PAYLOAD_TYPE:
        case 'image':
            print('a picture')
            # picture = predicted_msg[:-2].reshape(original_shape)

            # plt.figure()
            # plt.imshow(img_array)
            # # plt.show()

            # plt.figure()
            # plt.imshow(picture)
            # plt.show()

        case 'string':
            print('a string')
            print('Predicted message', predicted_msg)
            # print('Predicted message (Viterbi)', predicted_msg_viterbi)
            # print('Sent message (predicted by BCJR)', frombits(predicted_msg))

    num_errors = sum(abs(np.array(sent_message) - np.array(predicted_msg[:len(sent_message)])))
    error_percentage = 100 * num_errors / len(sent_message)
    error_ratio = error_percentage / 100
    BER_BCJR.append(error_ratio)

    # For some reason, Viterbi is shifted by one bit.
    predicted_msg_viterbi = predicted_msg_viterbi[1:1 + len(sent_message)]
    num_errors_viterbi = sum(abs(np.array(sent_message) - np.array(predicted_msg_viterbi)))
    error_percentage_viterbi = 100 * num_errors_viterbi / len(sent_message)
    error_ratio_viterbi = error_percentage_viterbi / 100
    BER_viterbi.append(error_ratio_viterbi)

    print(f'errors (BCJR): {num_errors} / {len(sent_message)} ({error_percentage:.3f} %, sigma={sigma:.2f})')
    # print(
    # f'errors (Viterbi): {num_errors_viterbi} / {len(sent_message)}
    # ({error_percentage_viterbi:.3f} %, sigma={sigma:.2f})')

    if PAYLOAD_TYPE == 'image' and round(N0) == 5:
        INTERPOLATION = None
        CMAP = "binary"
        if GREYSCALE:
            pixel_values = map_PPM_symbols(predicted_msg, 8)
            bcjr_picture = pixel_values[:original_shape[0] * original_shape[1]].reshape(original_shape)

            pixel_values = map_PPM_symbols(predicted_msg_viterbi, 8)
            viterbi_picture = pixel_values[:original_shape[0] * original_shape[1]].reshape(original_shape)
            CMAP = 'Greys'
            MODE = "L"
        else:
            bcjr_picture = predicted_msg[:original_shape[0] * original_shape[1]].reshape(original_shape)
            viterbi_picture = predicted_msg_viterbi[:original_shape[0] * original_shape[1]].reshape(original_shape)
            CMAP = 'binary'
            MODE = '1'

        if GREYSCALE:
            sent_picture = map_PPM_symbols(sent_message, 8).reshape(original_shape)
            noisy_picture = map_PPM_symbols(
                noisy_sent_message, 8).reshape(original_shape)
        else:
            sent_picture = sent_message.reshape(original_shape)
            noisy_picture = noisy_sent_message.reshape(original_shape)

        fig, axs = plt.subplots(2, 3)

        axs[0, 0].imshow(sent_picture, interpolation=INTERPOLATION, cmap=CMAP)
        axs[0, 0].set_title('Original')
        axs[0, 0].set_ylabel('Pixel number (y)')

        axs[0, 1].imshow(noisy_picture, cmap=CMAP)
        axs[0, 1].set_title(f'Original + AWGN (SNR={SNR[-1]:.1f} dB)')

        axs[0, 2].imshow(bcjr_picture, interpolation=INTERPOLATION, cmap=CMAP)
        axs[0, 2].set_title(f'BCJR decoded (BER={BER_BCJR[-1]:.3f})')

        axs[1, 0].imshow(sent_picture, interpolation=INTERPOLATION, cmap=CMAP)
        axs[1, 0].set_title('Original')
        axs[1, 0].set_ylabel('Pixel number (y)')
        axs[1, 0].set_xlabel('Pixel number (x)')

        axs[1, 1].imshow(noisy_picture, interpolation=INTERPOLATION, cmap=CMAP)
        axs[1, 1].set_title(f'Original + AWGN (SNR={SNR[-1]:.1f} dB)')
        axs[1, 1].set_xlabel('Pixel number (x)')

        axs[1, 2].imshow(viterbi_picture, interpolation=INTERPOLATION, cmap=CMAP)
        axs[1, 2].set_title(f'Viterbi decoded (BER={BER_viterbi[-1]:.3f})')
        axs[1, 2].set_xlabel('Pixel number (x)')

        plt.suptitle('BCJR and Viterbi decoding comparison')
        plt.show()

plt.figure()
plt.semilogy(SNR, BER_BCJR, label='BCJR')
plt.semilogy(SNR, BER_viterbi, label='Viterbi')

plt.legend()

plt.ylabel('BER (-)')
plt.xlabel('SNR (dB)')
plt.title('BER for the Viterbi & BCJR algorithm')

plt.show()

# pi_ak =

print('Done!')

print('Inner decoder')
# sent_message = np.array([0, 1, 0, 1, 0, 0])
