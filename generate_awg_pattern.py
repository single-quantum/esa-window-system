# %%
import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from BCJR_decoder_functions import ppm_symbols_to_bit_array, predict
from encoder_functions import (bit_interleave, channel_interleave, convolve,
                               get_csm, map_PPM_symbols, prepend_asm, slicer,
                               zero_terminate)
from parse_ppm_symbols import rolling_window
from ppm_parameters import (BIT_INTERLEAVE, CCSDS_ASM, CHANNEL_INTERLEAVE,
                            GREYSCALE, PAYLOAD_TYPE, M, bin_factor, m,
                            num_samples_per_slot, num_symbols_per_slice,
                            sample_size_awg, symbols_per_codeword)
from trellis import Trellis
from utils import AWGN, bpsk_encoding, generate_outer_code_edges, tobits

ADD_ASM: bool = True

num_output_bits: int = 3     # number of output bits from the convolutional encoder
num_input_bits: int = 1
memory_size: int = 2         # Memory size of the convolutional encoder

# 1 means: no guard slot, 5/4 means: M/4 guard slots
num_bins_per_symbol = int(bin_factor * M)

if GREYSCALE:
    IMG_MODE = "L"
else:
    IMG_MODE = "1"

match PAYLOAD_TYPE:
    case 'string':
        sent_message = np.array(tobits('Hello World'))
    case 'image':
        file = "JWST_2022-07-27_Jupiter_tiny.png"
        img = Image.open(file)
        img = img.convert(IMG_MODE)
        img_array = np.asarray(img).astype(int)

        plt.figure(figsize=(11, 10))
        plt.imshow(img_array, cmap="binary")
        plt.savefig('sent_image.png')
        plt.show()

        img_shape = img_array.shape
        # In the case of greyscale, each pixel has a value from 0 to 255.
        # This would be the same as saying that each pixel is a symbol, which should be mapped to an 8 bit sequence.
        if GREYSCALE:
            sent_message = ppm_symbols_to_bit_array(img_array.flatten(), 8)
        else:
            sent_message = img_array.flatten()

num_bits_sent = len(sent_message)


if PAYLOAD_TYPE != 'calibration':
    # Slice, prepend ASM and zero terminate each information block of 15120 bits.
    information_blocks = slicer(sent_message, include_crc=False)
    information_blocks = zero_terminate(information_blocks)

    while sent_message.shape[0] * num_output_bits / m != sent_message.shape[0] * num_output_bits // m:
        sent_message = np.append(sent_message, [0])

    # SCPPM encoder
    # The convolutional encoder is a 1/3 code rate encoder, so you end up with 3x more columns.
    convoluted_bit_sequence = np.zeros((information_blocks.shape[0], information_blocks.shape[1] * 3), dtype=int)
    for i, row in enumerate(information_blocks):
        convoluted_bit_sequence[i], _ = convolve(row)
        if BIT_INTERLEAVE:
            convoluted_bit_sequence[i] = bit_interleave(convoluted_bit_sequence[i])

    encoded_message = convoluted_bit_sequence.flatten()

    with open('jupiter_greyscale_64_samples_per_bin_interleaved_sent_bit_sequence', 'wb') as f:
        pickle.dump(encoded_message, f)

match PAYLOAD_TYPE:
    case 'calibration':
        sent_symbol = 0
        msg_PPM_symbols = [sent_symbol] * (num_symbols_per_slice - 1)
        # Zero terminate
        msg_PPM_symbols.append(0)
        num_slices = 1

    case _:
        num_PPM_symbols_before_interleaving = encoded_message.shape[0] // m
        msg_PPM_symbols = map_PPM_symbols(encoded_message, num_PPM_symbols_before_interleaving, m)

        B: int = 378
        N: int = 10
        assert (B * N) % symbols_per_codeword == 0, "The product of B and N should be a multiple of 15120/m"
        if CHANNEL_INTERLEAVE:
            msg_PPM_symbols = channel_interleave(msg_PPM_symbols, B, N)

        num_PPM_symbols = len(msg_PPM_symbols)

        # One SCPPM slice is 15120 symbols, as defined by the CCSDS protocol
        num_codewords = math.ceil(num_PPM_symbols / symbols_per_codeword)

# Insert CSMs
len_codeword = num_symbols_per_slice // m
num_codewords = msg_PPM_symbols.shape[0] // len_codeword
CSM = get_csm(M=M)

if ADD_ASM:
    ppm_mapped_message_with_csm = np.zeros(len(msg_PPM_symbols) + len(CSM) * num_codewords, dtype=int)
    for i in range(num_codewords):
        prepended_codeword = np.hstack((CSM, msg_PPM_symbols[i * len_codeword:(i + 1) * len_codeword]))
        ppm_mapped_message_with_csm[i * len(prepended_codeword):(i + 1) * len(prepended_codeword)] = prepended_codeword

    msg_PPM_symbols = ppm_mapped_message_with_csm


if len(msg_PPM_symbols) % 2 != 0:
    msg_PPM_symbols = np.append(msg_PPM_symbols, 0)


message_time_microseconds = sample_size_awg * 1E-12 * num_samples_per_slot * \
    num_bins_per_symbol * msg_PPM_symbols.shape[0] * 1E6

if PAYLOAD_TYPE == 'image':
    print(f'Sending image with shape {img_shape[0]}x{img_shape[1]}')
    with open('jupiter_tiny_sent_ppm_symbols', 'wb') as f:
        pickle.dump(msg_PPM_symbols, f)
print(f'num symbols sent: {msg_PPM_symbols.shape[0]}')
print(f'Number of codewords: {num_codewords}')
print(f'Datarate: {1000*num_bits_sent/(message_time_microseconds*1000):.2f} Mbps')
print(f'Message time span: {message_time_microseconds:.3f} microseconds')
print(f'Minimum window size needed: {2*message_time_microseconds:.3f} microseconds')

# %%

# Generate AWG pattern file
frame_width: int = num_bins_per_symbol * num_samples_per_slot
num_frames: int = len(msg_PPM_symbols)
pulse_width: int = 8

pulse = np.zeros(frame_width * num_frames)
print(f'Multiple of 256? {len(pulse)/256}')

for i, ppm_symbol_position in enumerate(msg_PPM_symbols):
    if ADD_ASM:
        idx = i * frame_width + ppm_symbol_position * num_samples_per_slot + num_samples_per_slot // 2 - pulse_width // 2
        pulse[idx:idx + pulse_width] = 30000
        continue

    # If no ASM is used, make the first peak a synchronisation peak
    if not ADD_ASM and i == 0 and ppm_symbol_position == 0:
        idx = i * frame_width + ppm_symbol_position * num_samples_per_slot + num_samples_per_slot // 2 - pulse_width // 2
        pulse[idx:idx + pulse_width] = 30000
    else:
        idx = i * frame_width + ppm_symbol_position * num_samples_per_slot + num_samples_per_slot // 2 - pulse_width // 2
        pulse[idx:idx + pulse_width] = 15000


# Add some zeros to more clearly distinguish between repeated messages.
pulse = np.hstack(([0] * frame_width * len(CCSDS_ASM) * 20, pulse))
pulse = np.hstack((pulse, [0] * frame_width * len(CCSDS_ASM) * 20))

# Convert to pandas dataframe for easy write to CSV
df = pd.DataFrame(pulse)

# %%
match PAYLOAD_TYPE:
    case 'image' if GREYSCALE:
        filepath = f'ppm_message_Jupiter_tiny_greyscale_{img_shape[0]}x{img_shape[1]}_slice_{num_samples_per_slot}_CSM_interleaved.csv'
        # filepath = f'test_message_greyscale_{img_shape[0]}x{img_shape[1]}_slice_{num_samples_per_slot}_CSM_not_bit_interleaved.csv'
        # filepath = 'test_message_all_zeros_interleaved.csv'
    case 'image' if not GREYSCALE:
        # filepath = f'ppm_message_Jupiter_tiny_{img_shape[0]}x{img_shape[1]}_slice_{num_samples_per_slot}_CSM_not_interleaved.csv'
        # filepath = f'test_message_{img_shape[0]}x{img_shape[1]}_slice_{num_samples_per_slot}_CSM_interleaved.csv'
        filepath = 'test_message_all_zeros.csv'

    case 'string':
        filepath = f'ppm_message_Hello_World_no_ASM.csv'
    case 'calibration':
        filepath = f'ppm_calibration_message_{len(msg_PPM_symbols)}_symbols_{num_samples_per_slot}_samples_per_slot_{sent_symbol}_CCSDS_ASM.csv'

print(f'Writing data to file {filepath}')
df.to_csv(filepath, index=False, header=None)
print(f'Wrote data to file {filepath}')
