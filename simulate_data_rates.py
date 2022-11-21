import math

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from encoder_functions import get_csm, slicer, zero_terminate
from ppm_parameters import (CHANNEL_INTERLEAVE, GREYSCALE, B_interleaver,
                            N_interleaver, bin_factor,
                            num_symbols_per_slice, sample_size_awg,
                            symbols_per_codeword)
from utils import print_ppm_parameters, save_figure

print_ppm_parameters()
ADD_ASM: bool = True

num_output_bits: int = 3     # number of output bits from the convolutional encoder


# 1 means: no guard slot, 5/4 means: M/4 guard slots

if GREYSCALE:
    IMG_MODE = "L"
else:
    IMG_MODE = "1"

num_symbols_interleaver = B_interleaver * N_interleaver * (N_interleaver - 1)

_num_bits_per_symbol = [i for i in range(2, 6)]
_samples_per_slot = [i for i in range(1, 6)]
top = []

for yi in _samples_per_slot:
    for xi in _num_bits_per_symbol:
        m = xi
        M = 2**xi
        num_bins_per_symbol = int(bin_factor * M)
        num_samples_per_slot = yi

        sent_message = np.zeros(int(1E5)).astype(int)
        num_bits_sent = len(sent_message)

        # Slice, prepend ASM and zero terminate each information block of 15120 bits.
        information_blocks = slicer(sent_message, include_crc=False)
        information_blocks = zero_terminate(information_blocks)

        # SCPPM encoder
        # The convolutional encoder is a 1/3 code rate encoder, so you end up with 3x more columns.
        convoluted_bit_sequence = np.zeros((information_blocks.shape[0], information_blocks.shape[1] * 3), dtype=int)

        encoded_message = convoluted_bit_sequence.flatten()

        num_PPM_symbols_before_interleaving = encoded_message.shape[0] // m
        msg_PPM_symbols = np.zeros(len(encoded_message) // m).astype(int)

        assert (B_interleaver * N_interleaver) % symbols_per_codeword == 0, "The product of B and N should be a multiple of 15120/m"
        if CHANNEL_INTERLEAVE:
            msg_PPM_symbols = np.zeros(len(msg_PPM_symbols) + B_interleaver * N_interleaver * (N_interleaver - 1))

        num_PPM_symbols = len(msg_PPM_symbols)

        # One SCPPM slice is 15120 symbols, as defined by the CCSDS protocol
        num_codewords = math.ceil(num_PPM_symbols / symbols_per_codeword)

        # Insert CSMs
        len_codeword = num_symbols_per_slice // m
        num_codewords = msg_PPM_symbols.shape[0] // len_codeword
        CSM = get_csm(M=M)

        if ADD_ASM:
            ppm_mapped_message_with_csm = np.zeros(len(msg_PPM_symbols) + len(CSM) * num_codewords, dtype=int)
            for j in range(num_codewords):
                prepended_codeword = np.hstack((CSM, msg_PPM_symbols[j * len_codeword:(j + 1) * len_codeword]))
                ppm_mapped_message_with_csm[j * len(prepended_codeword):(j + 1) *
                                            len(prepended_codeword)] = prepended_codeword

            msg_PPM_symbols = ppm_mapped_message_with_csm

        if len(msg_PPM_symbols) % 2 != 0:
            msg_PPM_symbols = np.append(msg_PPM_symbols, 0)

        message_time_microseconds = sample_size_awg * 1E-12 * num_samples_per_slot * \
            num_bins_per_symbol * msg_PPM_symbols.shape[0] * 1E6

        datarate = 1000 * num_bits_sent / (message_time_microseconds * 1000)
        top.append(datarate)

# setup the figure and axes
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111, projection='3d')

_xx, _yy = np.meshgrid(_num_bits_per_symbol, _samples_per_slot)
x, y = _xx.ravel(), _yy.ravel()

bottom = np.zeros_like(top)
width = 1
depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Data rates with 1/3 code rate')
ax1.set_xlabel('PPM order')
ax1.set_ylabel('Slot width (ps)')
ax1.set_zlabel('Data rate (Mpbs)')

ax1.set_yticks(_samples_per_slot)
ax1.set_yticklabels([str(112*i) for i in range(1, 6)])

ax1.set_xticks(_num_bits_per_symbol)
ax1.set_xticklabels([str(int(2**i)) for i in range(2, 6)])

plt.show()