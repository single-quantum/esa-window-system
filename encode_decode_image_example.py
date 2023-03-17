import numpy as np
from data_converter import message_from_payload
from scppm_encoder import encoder
from scppm_decoder import decode
from ppm_parameters import B_interleaver, N_interleaver, m, CODE_RATE, CSM, symbols_per_codeword
from fractions import Fraction
from encoder_functions import map_PPM_symbols
import matplotlib.pyplot as plt

from utils import flatten

# Convert image to bit array
sent_bits = message_from_payload('image', filepath='sample_payloads/pillars-of-creation-tiny.png')

# Put the payload through the encoder
slot_mapped_sequence = encoder(sent_bits)

# The decode message takes an array of PPM symbols, to the slot mapped message
# Should be converted to a ppm mapped message first.
ppm_mapped_message = np.nonzero(slot_mapped_sequence)[1]

# The ppm mapped message still includes the synchronisation marker. 
# Remove CSMs

ppm_mapped_message = ppm_mapped_message.reshape((-1, symbols_per_codeword+len(CSM)))
ppm_mapped_message = ppm_mapped_message[:, len(CSM):]

decoded_message = decode(ppm_mapped_message, B_interleaver, N_interleaver, m, CHANNEL_INTERLEAVE=True, BIT_INTERLEAVE=True, CODE_RATE=CODE_RATE)

pixel_values = map_PPM_symbols(decoded_message[0], 8)
img_arr = pixel_values[:100*101].reshape((101, 100))

plt.figure()
plt.imshow(img_arr)
plt.show()