import pickle
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from core.BCJR_decoder_functions import ppm_symbols_to_bit_array, predict
from core.encoder_functions import (bit_deinterleave, channel_deinterleave, get_csm,
                                    randomize, unpuncture)
from core.trellis import Trellis
from core.utils import bpsk_encoding, generate_outer_code_edges


class DecoderError(Exception):
    pass


def decode(
    slot_mapped_sequence: npt.NDArray[np.int_],
    M: int,
    CODE_RATE: Fraction,
    CHANNEL_INTERLEAVE=True,
    BIT_INTERLEAVE=True,
    **kwargs
) -> tuple[npt.NDArray[np.int_], float | None]:
    user_settings = kwargs.get('user_settings', {})

    # The decode message takes an array of PPM symbols, so the slot mapped message
    # Should be converted to a ppm mapped message first.
    ppm_mapped_message = np.nonzero(slot_mapped_sequence)[1]

    # The ppm mapped message still includes the synchronisation marker.
    # Remove CSMs
    CSM = get_csm(M)
    m = int(np.log2(M))
    symbols_per_codeword: int = int(15120 / m)

    ppm_mapped_message = ppm_mapped_message.reshape((-1, symbols_per_codeword + len(CSM)))
    ppm_mapped_message = ppm_mapped_message[:, len(CSM):]
    ppm_mapped_message = ppm_mapped_message.flatten()

    convoluted_bit_sequence: npt.NDArray[np.int_]

    # Deinterleave
    if CHANNEL_INTERLEAVE:
        B_interleaver = user_settings.get('B_interleaver')
        N_interleaver = user_settings.get('N_interleaver', 2)
        if B_interleaver is None:
            m = int(np.log2(M))
            B_interleaver = int(15120 / m / N_interleaver)

        print('Deinterleaving PPM symbols')
        ppm_mapped_message = channel_deinterleave(ppm_mapped_message, B_interleaver, N_interleaver)
        num_zeros_interleaver: int = (2 * B_interleaver * N_interleaver * (N_interleaver - 1))
        convoluted_bit_sequence = ppm_symbols_to_bit_array(
            ppm_mapped_message[:(len(ppm_mapped_message) - num_zeros_interleaver)], m)
    else:
        convoluted_bit_sequence = ppm_symbols_to_bit_array(ppm_mapped_message, m)

    BER_before_decoding: float | None = None

    # Get the BER before decoding
    if reference_file_path := user_settings.get('reference_file_path'):
        with open(reference_file_path, 'rb') as f:
            sent_bit_sequence: list = pickle.load(f)

        if len(convoluted_bit_sequence) > len(sent_bit_sequence):
            BER_before_decoding = np.sum(np.abs(convoluted_bit_sequence[:len(sent_bit_sequence)] -
                                                sent_bit_sequence)) / len(sent_bit_sequence)
        else:
            num_wrong_bits = np.sum(
                np.abs(convoluted_bit_sequence - sent_bit_sequence[:len(convoluted_bit_sequence)])
            )
            BER_before_decoding = num_wrong_bits / len(sent_bit_sequence)

        print(f'BER before decoding: {BER_before_decoding}')
        # if BER_before_decoding > 0.25:
        #     raise DecoderError("Could not properly decode message. ")

    num_leftover_symbols = convoluted_bit_sequence.shape[0] % 15120
    if (diff := 15120 - num_leftover_symbols) < 100:
        convoluted_bit_sequence = np.hstack((convoluted_bit_sequence, np.zeros(diff, dtype=int)))
        num_leftover_symbols = convoluted_bit_sequence.shape[0] % 15120

    symbols_to_deinterleave = convoluted_bit_sequence.shape[0] - num_leftover_symbols

    received_sequence_interleaved = convoluted_bit_sequence[:symbols_to_deinterleave].reshape((-1, 15120))

    if BIT_INTERLEAVE:
        print('Bit deinterleaving')
        received_sequence = np.zeros_like(received_sequence_interleaved)
        for i, row in enumerate(received_sequence_interleaved):
            received_sequence[i] = bit_deinterleave(row)
    else:
        received_sequence = received_sequence_interleaved

    deinterleaved_received_sequence = received_sequence.flatten()

    print('Setting up trellis')

    # Trellis paramters (can be defined outside of for loop for optimisation)
    num_output_bits: int = 3
    num_input_bits: int = 1
    memory_size: int = 2
    edges = generate_outer_code_edges(memory_size, bpsk_encoding=False)

    time_steps = int(deinterleaved_received_sequence.shape[0] * float(CODE_RATE))

    if kwargs.get('use_cached_trellis'):
        cached_trellis_file_path = kwargs['cached_trellis_file_path']
        cached_trellis = kwargs['cached_trellis']
        if time_steps == 80640 and cached_trellis_file_path.is_file():
            tr = cached_trellis
        else:
            tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
            tr.set_edges(edges)
    else:
        tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
        tr.set_edges(edges)

    Es = 5

    encoded_sequence = bpsk_encoding(deinterleaved_received_sequence.astype(float))

    encoded_sequence = unpuncture(encoded_sequence, CODE_RATE)

    predicted_msg: npt.NDArray[np.int_] = predict(tr, encoded_sequence, Es=Es)
    information_block_sizes = {
        Fraction(1, 3): 5040,
        Fraction(1, 2): 7560,
        Fraction(2, 3): 10080
    }

    num_bits = information_block_sizes[CODE_RATE]
    information_blocks: npt.NDArray[np.int_] = predicted_msg.reshape((-1, num_bits))[:, :-2].flatten()

    # information_blocks = predicted_msg.reshape((-1, 5040)).flatten()
    # Derandomize
    # information_blocks = randomize(information_blocks)

    while information_blocks.shape[0] / 8 != information_blocks.shape[0] // 8:
        information_blocks = np.hstack((information_blocks, 0))

    return information_blocks, BER_before_decoding
