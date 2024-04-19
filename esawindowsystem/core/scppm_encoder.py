import pickle
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from esawindowsystem.core.encoder_functions import (accumulate, bit_interleave,
                                                    channel_interleave, convolve, get_csm,
                                                    map_PPM_symbols, puncture, randomize,
                                                    slicer, slot_map, zero_terminate)

from esawindowsystem.core.utils import ppm_symbols_to_bit_array


def preprocess_bit_stream(bit_stream: npt.NDArray[np.int_], code_rate: Fraction, **kwargs) -> npt.NDArray[np.int_]:
    """This preprocessing function slices the bit stream in information blocks and attaches the CRC. """
    # Slice into information blocks of 5038 bits (code rate 1/3) and append 2 termination bits.
    # CRC attachment is still to be implemented
    information_blocks = slicer(bit_stream, code_rate, include_crc=False)
    with open('sent_bit_sequence_no_csm', 'wb') as f:
        pickle.dump(information_blocks.flatten(), f)

    if kwargs.get('use_randomizer', False):
        information_blocks = randomize(information_blocks)
    information_blocks = zero_terminate(information_blocks)

    return information_blocks


def SCPPM_encoder(
    information_blocks: npt.NDArray,
    M: int,
    code_rate: Fraction,
    BIT_INTERLEAVE: bool = True,
    **kwargs
):
    """The SCPPM encoder consists of the convolutional encoder, code interleaver, accumulator and PPM symbol mapper.

    Returns a sequence of PPM symbols.
    """

    # The convolutional encoder is a 1/3 code rate encoder, so you end up with
    # 3x more columns.
    convoluted_bit_sequence = np.zeros(
        (information_blocks.shape[0], information_blocks.shape[1] * 3), dtype=int)

    for i, row in enumerate(information_blocks):
        convoluted_bit_sequence[i], _ = convolve(row)

    if code_rate != Fraction(1, 3):
        convolutional_codewords: npt.NDArray = puncture(convoluted_bit_sequence, code_rate)
    else:
        convolutional_codewords = convoluted_bit_sequence

    if BIT_INTERLEAVE:
        for i, row in enumerate(convolutional_codewords):
            convolutional_codewords[i] = bit_interleave(convolutional_codewords[i])

    if kwargs.get('use_inner_encoder'):
        for i, row in enumerate(convolutional_codewords):
            convolutional_codewords[i] = accumulate(convolutional_codewords[i])

    encoded_message = convolutional_codewords.flatten()

    # The encoded message can be saved to a file, to compare the BER before
    # and after decoding
    save_encoded_sequence_to_file = kwargs.get('save_encoded_sequence_to_file', False)

    if save_encoded_sequence_to_file:
        reference_file_prefix: str = kwargs.get('reference_file_prefix', 'sample_payload')
        num_samples_per_slot: int | None = kwargs.get('num_samples_per_slot')

        filename: str = f'{reference_file_prefix}_{num_samples_per_slot}_samples_per_slot_{M}' +\
            '-PPM_interleaved_sent_bit_sequence'
        with open(filename, 'wb') as f:
            pickle.dump(encoded_message, f)

    # Map the encoded message bit stream to PPM symbols
    m: int = int(np.log2(M))
    msg_PPM_symbols = map_PPM_symbols(encoded_message, m)

    return msg_PPM_symbols


def postprocess_ppm_symbols(
    PPM_symbols,
    M: int,
    B_interleaver: int,
    N_interleaver: int,
    CHANNEL_INTERLEAVE: bool = True,
    q: int = 1,
    **kwargs
):
    """Takes the PPM symbols and interleaves them, adds the CSM and repeats the message q times. """
    # Note: repeater not yet implemented.
    if CHANNEL_INTERLEAVE:
        PPM_symbols = channel_interleave(PPM_symbols, B_interleaver, N_interleaver)

    symbols_per_codeword = int(15120 / np.log2(M))
    num_codewords = int(PPM_symbols.shape[0] / symbols_per_codeword)

    # # Attach Codeword Synchronisation Markerk (CSM) to each codeword of
    # # 15120/m PPM symbols
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


def encoder(
        bit_stream: npt.NDArray[np.int_],
        M: int,
        code_rate: Fraction,
        **kwargs) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """Does some preprocessing steps to the bit_stream (slice bit stream into blocks, add CRC), puts it through the SCPPM_encoder and post-processing (interleave, add CSM).

    Returns a slot mapped binary vector.
    """

    user_settings: dict = kwargs.get('user_settings', {})
    # try:
    #     check_user_settings(user_settings)
    # except KeyError as e:
    #     print(e)
    #     raise KeyError("User settings is missing required parameters. ")

    B_interleaver: int | None = user_settings.get('B_interleaver')
    # If no number of parallel shift registers is defined, use the minimum of 2
    N_interleaver: int = user_settings.get('N_interleaver', 2)

    if B_interleaver is None:
        m = int(np.log2(M))
        B_interleaver = int(15120 / m / N_interleaver)

    information_blocks = preprocess_bit_stream(bit_stream, code_rate, **kwargs)
    PPM_symbols = SCPPM_encoder(information_blocks, M, code_rate, **kwargs)

    slot_mapped_sequence = postprocess_ppm_symbols(
        PPM_symbols, M, B_interleaver, N_interleaver
    )

    with open('sent_bit_sequence', 'wb') as f:
        sent_ppm_symbols = np.nonzero(slot_mapped_sequence)[1]
        sent_bit_sequence = ppm_symbols_to_bit_array(sent_ppm_symbols, int(np.log2(M)))
        pickle.dump(sent_bit_sequence, f)

    return slot_mapped_sequence, sent_bit_sequence, information_blocks
