# %%
import math

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from esawindowsystem.core.encoder_functions import get_csm, slicer, zero_terminate
from esawindowsystem.ppm_parameters import (CHANNEL_INTERLEAVE, GREYSCALE, B_interleaver, M,
                            N_interleaver, bin_factor, m, num_samples_per_slot,
                            num_symbols_per_slice, sample_size_awg,
                            symbols_per_codeword)
from esawindowsystem.core.utils import print_ppm_parameters, save_figure

print_ppm_parameters()
ADD_ASM: bool = True

num_output_bits: int = 3     # number of output bits from the convolutional encoder


# 1 means: no guard slot, 5/4 means: M/4 guard slots
num_bins_per_symbol = int(bin_factor * M)

if GREYSCALE:
    IMG_MODE = "L"
else:
    IMG_MODE = "1"

x = np.array([i * 10000 for i in range(5, 100)])

num_symbols_interleaver = B_interleaver * N_interleaver * (N_interleaver - 1)
y1 = 100 * num_symbols_interleaver / x
y2 = []

for i in x:
    sent_message = np.zeros(int(i)).astype(int)
    num_bits_sent = len(sent_message)

    # Slice, prepend ASM and zero terminate each information block of 15120 bits.
    information_blocks = slicer(sent_message, include_crc=False)
    information_blocks = zero_terminate(information_blocks)

    # SCPPM encoder
    # The convolutional encoder is a 1/3 code rate encoder, so you end up with 3x more columns.
    convoluted_bit_sequence = np.zeros((information_blocks.shape[0], information_blocks.shape[1] * 3), dtype=int)

    encoded_message = convoluted_bit_sequence.flatten()

    num_PPM_symbols_before_interleaving = encoded_message.shape[0] // m
    # msg_PPM_symbols = map_PPM_symbols(encoded_message, num_PPM_symbols_before_interleaving, m)
    msg_PPM_symbols = np.zeros(len(encoded_message) // m).astype(int)

    assert (B_interleaver * N_interleaver) % symbols_per_codeword == 0, "The product of B and N should be a multiple of 15120/m"
    if CHANNEL_INTERLEAVE:
        # msg_PPM_symbols = channel_interleave(msg_PPM_symbols, B, N)
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
    y2.append(datarate)

data = zip(x * 1E-6, y1, y2)

print(tabulate(data, headers=['Bits sent (Mbits)', 'Interleaver overhead (%)', 'Data rate (Mbps)']))

fig, axs1 = plt.subplots()
axs1.set_title('Channel interleaver overhead')
line1, = axs1.plot(x / 1E6, y1, label='Interleaver overhead')
axs1.set_xlabel('Bits sent (Mbits)')
axs1.set_ylabel('% interleaver overhead')


axs2 = axs1.twinx()
axs2.set_ylabel('Data rate (Mbps)')
line2, = axs2.plot(x / 1E6, y2, color='r', label='Datarate')

axs2.legend(handles=[line1, line2], loc='center right')
save_figure(plt, 'interleaver-overhead.png', 'simulate_channel_interleaver_overhead')
plt.show()
