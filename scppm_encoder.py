import pickle

import numpy as np

from encoder_functions import (bit_interleave, channel_interleave, convolve,
                               get_csm, map_PPM_symbols, randomize, slicer,
                               slot_map, zero_terminate)
from ppm_parameters import (BIT_INTERLEAVE, CHANNEL_INTERLEAVE, B_interleaver,
                            M, N_interleaver, m, num_samples_per_slot,
                            symbols_per_codeword)


def preprocess_bit_stream(bit_stream):
    """This preprocessing function slices the bit stream in information blocks and attaches the CRC. """
    # Slice into information blocks of 5038 bits (code rate 1/3) and append 2 termination bits.
    # CRC attachment is still to be implemented
    information_blocks = slicer(bit_stream, include_crc=False)
    information_blocks = zero_terminate(information_blocks)
    information_blocks = randomize(information_blocks)

    return information_blocks


def SCPPM_encoder(information_blocks, save_encoded_sequence_to_file=True):
    """The SCPPM encoder consists of the convolutional encoder, code interleaver, accumulator and PPM symbol mapper.

    Returns a sequence of PPM symbols.
    """
    # The convolutional encoder is a 1/3 code rate encoder, so you end up with
    # 3x more columns.
    convoluted_bit_sequence = np.zeros(
        (information_blocks.shape[0], information_blocks.shape[1] * 3), dtype=int)

    for i, row in enumerate(information_blocks):
        convoluted_bit_sequence[i], _ = convolve(row)
        if BIT_INTERLEAVE:
            convoluted_bit_sequence[i] = bit_interleave(
                convoluted_bit_sequence[i])

    encoded_message = convoluted_bit_sequence.flatten()

    # The encoded message can be saved to a file, to compare the BER before
    # and after decoding
    if save_encoded_sequence_to_file:
        with open(f'jupiter_greyscale_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence', 'wb') as f:
            pickle.dump(encoded_message, f)

    # Map the encoded message bit stream to PPM symbols
    msg_PPM_symbols = map_PPM_symbols(encoded_message, m)

    return msg_PPM_symbols


def postprocess_ppm_symbols(PPM_symbols, q=1):
    """Takes the PPM symbols and interleaves them, adds the CSM and repeats the message q times. """
    # Note: repeater not yet implemented.
    if CHANNEL_INTERLEAVE:
        PPM_symbols = channel_interleave(
            PPM_symbols, B_interleaver, N_interleaver)

    num_codewords = PPM_symbols.shape[0] // symbols_per_codeword

    # Attach Codeword Synchronisation Markerk (CSM) to each codeword of
    # 15120/m PPM symbols
    CSM = get_csm(M=M)

    ppm_mapped_message_with_csm = np.zeros(
        len(PPM_symbols) + len(CSM) * num_codewords, dtype=int)
    for i in range(num_codewords):
        prepended_codeword = np.hstack(
            (CSM, PPM_symbols[i * symbols_per_codeword:(i + 1) * symbols_per_codeword]))
        ppm_mapped_message_with_csm[
            i * len(prepended_codeword):(i + 1) * len(prepended_codeword)
        ] = prepended_codeword

    PPM_symbols = ppm_mapped_message_with_csm

    slot_mapped_sequence = slot_map(PPM_symbols, M)

    return slot_mapped_sequence


def encoder(bit_stream):
    """Does some preprocessing to the bit_stream, puts it through the SCPPM_encoder and does some post-processing.

    Returns a slot mapped binary vector.
    """

    information_blocks = preprocess_bit_stream(bit_stream)
    PPM_symbols = SCPPM_encoder(information_blocks)
    slot_mapped_sequence = postprocess_ppm_symbols(PPM_symbols)

    return slot_mapped_sequence
