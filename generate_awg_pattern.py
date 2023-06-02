# %%
import math

import numpy as np
import numpy.typing as npt
import pandas as pd

from data_converter import payload_to_bit_sequence
from encoder_functions import slot_map
from ppm_parameters import (BIT_INTERLEAVE, CHANNEL_INTERLEAVE, CODE_RATE, CSM, IMG_FILE_PATH,
                            GREYSCALE, IMG_SHAPE, PAYLOAD_TYPE, M, slot_length,
                            m, num_samples_per_slot, num_symbols_per_slice,
                            sample_size_awg, num_slots_per_symbol, symbols_per_codeword, B_interleaver, N_interleaver)
from scppm_encoder import encoder


pulse_width: int = 3        # number of DAC samples for 1 symbol
ADD_ASM: bool = True

msg_PPM_symbols: npt.NDArray[np.int_] = np.array([])
num_PPM_symbols: int
num_bits_sent: int
slot_mapped_sequence: npt.NDArray[np.int_]
sent_symbol: int | None = None

match PAYLOAD_TYPE:
    case 'calibration':
        sent_symbol = 1
        msg_PPM_symbols = np.array([sent_symbol] * (num_symbols_per_slice - 1))
        # Zero terminate
        msg_PPM_symbols = np.append(msg_PPM_symbols, 0)
        num_slices: int = 1

        # Insert CSMs
        len_codeword: int = num_symbols_per_slice // m
        num_codewords: int = msg_PPM_symbols.shape[0] // len_codeword

        if ADD_ASM:
            ppm_mapped_message_with_csm = np.zeros(len(msg_PPM_symbols) + len(CSM) * num_codewords, dtype=int)
            for i in range(num_codewords):
                prepended_codeword = np.hstack((CSM, msg_PPM_symbols[i * len_codeword:(i + 1) * len_codeword]))
                ppm_mapped_message_with_csm[i * len(prepended_codeword):(i + 1) *
                                            len(prepended_codeword)] = prepended_codeword

            msg_PPM_symbols = ppm_mapped_message_with_csm

        if len(msg_PPM_symbols) % 2 != 0:
            msg_PPM_symbols = np.append(msg_PPM_symbols, 0)

        message_time_microseconds = sample_size_awg * 1E-12 * num_samples_per_slot * \
            num_slots_per_symbol * msg_PPM_symbols.shape[0] * 1E6
        num_PPM_symbols = msg_PPM_symbols.shape[0]
        num_bits_sent = num_PPM_symbols * m
        slot_mapped_sequence = slot_map(msg_PPM_symbols, M)

    case _:
        sent_message: npt.NDArray[np.int_] = payload_to_bit_sequence(
            PAYLOAD_TYPE, filepath=IMG_FILE_PATH)
        num_bits_sent = len(sent_message)

        slot_mapped_sequence = encoder(sent_message, M, CODE_RATE,
                                       **{'user_settings': {
                                           'B_interleaver': B_interleaver,
                                           'N_interleaver': N_interleaver},
                                           'save_encoded_sequence_to_file': True,
                                           'reference_file_prefix': 'jupiter_greyscale',
                                           'num_samples_per_slot': num_samples_per_slot}
                                       )
        num_PPM_symbols = slot_mapped_sequence.shape[0]

        # One SCPPM codeword is 15120/m symbols, as defined by the CCSDS protocol
        num_codewords = math.ceil(num_PPM_symbols / (symbols_per_codeword + len(CSM)))
        num_slots = slot_mapped_sequence.flatten().shape[0]
        message_time_microseconds = num_slots * slot_length * 1E6


if PAYLOAD_TYPE == 'image':
    print(f'Sending image with shape {IMG_SHAPE[0]}x{IMG_SHAPE[1]}')
print(f'num symbols sent: {num_PPM_symbols}')
print(f'Number of symbols per second: {1/(message_time_microseconds*1E-6)*num_PPM_symbols}')
print(f'Number of codewords: {num_codewords}')
print(f'Datarate: {1000*num_bits_sent/(message_time_microseconds*1000):.2f} Mbps')
print(f'Message time span: {message_time_microseconds:.3f} microseconds')
print(f'Minimum window size needed: {2*message_time_microseconds:.3f} microseconds')

# %%

# Generate AWG pattern file
num_samples_per_symbol: int = num_slots_per_symbol * num_samples_per_slot

pulse = np.zeros(num_samples_per_symbol * num_PPM_symbols)
print(f'Multiple of 256? {len(pulse)/256}')

for i, slot_mapped_symbol in enumerate(slot_mapped_sequence):
    ppm_symbol_position = slot_mapped_symbol.nonzero()[0][0]
    if ADD_ASM:
        idx = i * num_samples_per_symbol + ppm_symbol_position * \
            num_samples_per_slot + num_samples_per_slot // 2 - pulse_width // 2
        pulse[idx:idx + pulse_width] = 30000
        continue

    # If no ASM is used, make the first peak a synchronisation peak
    if not ADD_ASM and i == 0 and ppm_symbol_position == 0:
        idx = i * num_samples_per_symbol + ppm_symbol_position * \
            num_samples_per_slot + num_samples_per_slot // 2 - pulse_width // 2
        pulse[idx:idx + pulse_width] = 30000
    else:
        idx = i * num_samples_per_symbol + ppm_symbol_position * \
            num_samples_per_slot + num_samples_per_slot // 2 - pulse_width // 2
        pulse[idx:idx + pulse_width] = 15000


# Add some zeros to more clearly distinguish between repeated messages.
pulse = np.hstack(([0] * num_samples_per_symbol * len(CSM) * 20, pulse))
pulse = np.hstack((pulse, [0] * num_samples_per_symbol * len(CSM) * 20))

# Convert to pandas dataframe for easy write to CSV
df = pd.DataFrame(pulse)

# %%
interleave_code = f'c{int(CHANNEL_INTERLEAVE)}b{int(BIT_INTERLEAVE)}'

# '/' is not allowed in filenames.
cr = str(CODE_RATE).replace('/', '-')

filepath: str = ''

match PAYLOAD_TYPE:
    case 'image' if GREYSCALE:
        filepath = f'ppm_message_Jupiter_tiny_greyscale_{IMG_SHAPE[0]}x{IMG_SHAPE[1]}_pixels_' +\
            f'{M}-PPM_{num_samples_per_slot}_{pulse_width}_{interleave_code}_{cr}-code-rate.csv'
    case 'image' if not GREYSCALE:
        filepath = f'ppm_message_Jupiter_tiny_greyscale_{IMG_SHAPE[0]}x{IMG_SHAPE[1]}_pixels_' +\
            f'{M}-PPM_{num_samples_per_slot}_{pulse_width}_b1c1_{cr}-code-rate.csv'
    case 'string':
        filepath = 'ppm_message_Hello_World_no_ASM.csv'
    case 'calibration':
        filepath = f'ppm_calibration_message_{len(msg_PPM_symbols)}_' +\
            f'symbols_{num_samples_per_slot}_samples_per_slot_{sent_symbol}_CCSDS_ASM.csv'
    case _:
        raise ValueError("Payload type not recognized. Should be one of ['image', 'string', 'calibration']")

print(f'Writing data to file {filepath}')
df.to_csv(filepath, index=False, header=False)
print(f'Wrote data to file {filepath}')
