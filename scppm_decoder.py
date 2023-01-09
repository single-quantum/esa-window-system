import pickle
from fractions import Fraction

import numpy as np

from BCJR_decoder_functions import ppm_symbols_to_bit_array, predict
from encoder_functions import bit_deinterleave, channel_deinterleave, randomize
from trellis import Trellis
from utils import bpsk_encoding, generate_outer_code_edges


class DecoderError(Exception):
    pass


def decode(ppm_mapped_message, B_interleaver, N_interleaver, m, CHANNEL_INTERLEAVE=True,
           BIT_INTERLEAVE=True, CODE_RATE=Fraction(1, 3), **kwargs):
    # Deinterleave
    if CHANNEL_INTERLEAVE:
        print('Deinterleaving PPM symbols')
        ppm_mapped_message = channel_deinterleave(ppm_mapped_message, B_interleaver, N_interleaver)
        num_zeros_interleaver = (2 * B_interleaver * N_interleaver * (N_interleaver - 1))
        convoluted_bit_sequence = ppm_symbols_to_bit_array(
            ppm_mapped_message[:(len(ppm_mapped_message) - num_zeros_interleaver)], m)
    else:
        convoluted_bit_sequence = ppm_symbols_to_bit_array(ppm_mapped_message, m)

    # Get the BER before decoding
    with open('jupiter_greyscale_8_samples_per_slot_8-PPM_interleaved_sent_bit_sequence', 'rb') as f:
        sent_bit_sequence: list = pickle.load(f)

    if len(convoluted_bit_sequence) > len(sent_bit_sequence):
        BER_before_decoding = np.sum(np.abs(convoluted_bit_sequence[:len(sent_bit_sequence)] -
                                            sent_bit_sequence)) / len(sent_bit_sequence)
    else:
        BER_before_decoding = np.sum(np.abs(convoluted_bit_sequence -
                                            sent_bit_sequence[:len(convoluted_bit_sequence)])) / len(sent_bit_sequence)

    print(f'BER before decoding: {BER_before_decoding}')
    if BER_before_decoding > 0.25:
        raise DecoderError("Could not properly decode message. ")
        # continue

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

    deinterleaved_received_sequence_2 = received_sequence.flatten()
    # deinterleaved_received_sequence = np.hstack((deinterleaved_received_sequence, [0, 0]))

    print('Setting up trellis')

    # Trellis paramters (can be defined outside of for loop for optimisation)
    num_output_bits: int = 3
    num_input_bits: int = 1
    memory_size: int = 2
    edges = generate_outer_code_edges(memory_size, bpsk_encoding=False)

    time_steps = int(deinterleaved_received_sequence_2.shape[0] * float(CODE_RATE))

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
        # df = 0

        # if df == 0 and z == 0:
        #     with open(f'cached_trellis_{time_steps}_timesteps', 'wb') as f:
        #         pickle.dump(tr, f)

    Es = 5
    N0 = 1

    encoded_sequence_2 = bpsk_encoding(deinterleaved_received_sequence_2.astype(float))

    predicted_msg = predict(tr, encoded_sequence_2, Es=Es)
    termination_bits_removed = predicted_msg.reshape((-1, 5040))[:, :-2].flatten()
    # Derandomize
    information_blocks = randomize(termination_bits_removed)

    while information_blocks.shape[0] / 8 != information_blocks.shape[0] // 8:
        information_blocks = np.hstack((information_blocks, 0))

    return information_blocks, BER_before_decoding
