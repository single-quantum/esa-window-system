from fractions import Fraction

import matplotlib.pyplot as plt
from PIL import Image

from data_converter import message_from_payload
from encoder_functions import map_PPM_symbols
from scppm_decoder import decode
from scppm_encoder import encoder

# Definitions:
# - M: PPM order, should be increment of 2**m with m in [2, 3, 4, ...]
# - CODE_RATE: code rate of the encoder, should be one of [1/3, 2/3, 1/2]
# - B_interleaver: base length of the shift register of the channel interleaver
# - N_interleaver: number of parallel shift registers in the channel interleaver
#   Note: the product B*N should be a multiple of 15120/m with m np.log2(M)
# - reference_file_path: file path to the pickle file of the encoded bit sequence.
#   This file is used to determine the bit error ratio (BER) before decoding.

M: int = 8
code_rate: Fraction = Fraction(2, 3)
payload_type = 'image'
payload_file_path = 'sample_payloads/pillars-of-creation-tiny.png'

IMG_SIZE: tuple[int, ...] = tuple((0, 0))

if payload_type == 'image':
    img = Image.open(payload_file_path)
    IMG_SIZE = img.size

# Additional settings that are not strictly necessary.
# If B and N are not provided, N is assumed to be 2
user_settings = {
    'B_interleaver': 2520,
    'N_interleaver': 2,
    'reference_file_path': 'pillars_greyscale_16_samples_per_slot_8-PPM_interleaved_sent_bit_sequence'
}

# 1. Convert payload to bit sequence
# 2. Encode
# 3. Decode

# Convert payload (in this case an image) to bit array
sent_bits = message_from_payload(payload_type, filepath=payload_file_path)

# Put the payload through the encoder
slot_mapped_sequence = encoder(
    sent_bits,
    M,
    code_rate,
    **{
        'user_settings': user_settings,
        'save_encoded_sequence_to_file': True,
        'reference_file_prefix': 'pillars_greyscale',
        'num_samples_per_slot': 16
    })

decoded_message = decode(slot_mapped_sequence, M, code_rate, **{'user_settings': user_settings})

if payload_type == 'image':
    pixel_values = map_PPM_symbols(decoded_message[0], 8)
    img_arr = pixel_values[:IMG_SIZE[0] * IMG_SIZE[1]].reshape((IMG_SIZE[1], IMG_SIZE[0]))

    plt.figure()
    plt.imshow(img_arr)
    plt.show()

else:
    print('Decoded message', decoded_message)
