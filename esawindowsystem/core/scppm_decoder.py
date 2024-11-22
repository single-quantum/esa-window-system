import pickle
from fractions import Fraction
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from esawindowsystem.core.BCJR_decoder_functions import (
    pi_ck, ppm_symbols_to_bit_array, predict, predict_iteratively)
from esawindowsystem.core.encoder_functions import (bit_deinterleave,
                                                    channel_deinterleave,
                                                    get_asm_bit_arr, get_csm,
                                                    randomize, slot_map,
                                                    unpuncture)
from esawindowsystem.core.trellis import Trellis
from esawindowsystem.core.utils import (bpsk_encoding,
                                        generate_outer_code_edges,
                                        get_BER_before_decoding, poisson_noise)


class DecoderError(Exception):
    pass


def decode(
    slot_mapped_sequence: npt.NDArray[np.int_],
    M: int,
    CODE_RATE: Fraction,
    CHANNEL_INTERLEAVE: bool = True,
    BIT_INTERLEAVE: bool = True,
    use_inner_encoder: bool = False,
    **kwargs: dict[str, Any]
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
    B_interleaver = user_settings.get('B_interleaver')
    N_interleaver = user_settings.get('N_interleaver', 2)
    if B_interleaver is None:
        m = int(np.log2(M))
        B_interleaver = int(15120 / m / N_interleaver)

    deinterleaved_ppm_symbols = channel_deinterleave(ppm_mapped_message, B_interleaver, N_interleaver)
    num_zeros_interleaver: int = (2 * B_interleaver * N_interleaver * (N_interleaver - 1))
    deinterleaved_slot_mapped_sequence = slot_map(deinterleaved_ppm_symbols[:(len(
        deinterleaved_ppm_symbols) - num_zeros_interleaver)], M, insert_guardslots=False)

    if CHANNEL_INTERLEAVE:

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
        BER_before_decoding = get_BER_before_decoding(reference_file_path, convoluted_bit_sequence)

        print(f'BER before decoding: {BER_before_decoding}')

    # Double check if this is still needed
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
    print()

    num_output_bits: int = 3
    num_input_bits: int = 1
    memory_size: int = 2
    edges = generate_outer_code_edges(memory_size, bpsk_encoding=False)

    time_steps = int(deinterleaved_received_sequence.shape[0] * float(CODE_RATE))

    if not use_inner_encoder:
        if kwargs.get('use_cached_trellis'):
            cached_trellis_file_path = kwargs.get('cached_trellis_file_path')
            # Note to self: do this more proper, now it just falls back silently if the file path is not found.
            if cached_trellis_file_path is not None and cached_trellis_file_path.is_file():
                with open(cached_trellis_file_path, 'rb') as f:
                    cached_trellis = pickle.load(f)
                    tr = cached_trellis
            else:
                tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
                tr.set_edges(edges)

                with open(f'cached_trellis_{time_steps}_timesteps', 'wb') as f:
                    pickle.dump(tr, f)

        else:
            tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
            tr.set_edges(edges)

        Es = 5

        encoded_sequence = bpsk_encoding(deinterleaved_received_sequence.astype(float))
        encoded_sequence = unpuncture(encoded_sequence, CODE_RATE)
        predicted_msg: npt.NDArray[np.int_] = predict(tr, encoded_sequence, Es=Es)
    else:
        predicted_msg, _, _ = predict_iteratively(deinterleaved_slot_mapped_sequence, M,
                                                  CODE_RATE, max_num_iterations=3, **kwargs)
    information_block_sizes = {
        Fraction(1, 3): 5040,
        Fraction(1, 2): 7560,
        Fraction(2, 3): 10080
    }

    num_bits = information_block_sizes[CODE_RATE]
    include_CRC = False
    if include_CRC:
        information_blocks: npt.NDArray[np.int_] = predicted_msg.reshape((-1, num_bits))[:, :-34].flatten()
    else:
        information_blocks: npt.NDArray[np.int_] = predicted_msg.reshape((-1, num_bits))[:, :-2].flatten()

    # information_blocks = predicted_msg.reshape((-1, 5040)).flatten()
    # Derandomize
    if not use_inner_encoder and kwargs.get('use_randomizer', False):
        information_blocks = randomize(information_blocks.reshape((-1, num_bits - 2)))
        information_blocks = information_blocks.flatten()

    while information_blocks.shape[0] / 8 != information_blocks.shape[0] // 8:
        information_blocks = np.hstack((information_blocks, 0))

    # For now, assume there is only one 32-bit ASM and remove it.
    ASM_arr = get_asm_bit_arr()

    asm_corr = np.correlate(information_blocks, ASM_arr, 'valid')

    if kwargs.get('debug_mode'):
        plt.figure()
        plt.plot(asm_corr)
        plt.show()

        plt.figure()
        plt.close()

    where_asms = np.where(asm_corr >= 18.0)[0]

    if where_asms.shape[0] == 0:
        raise DecoderError('ASM not found in message')

    # TODO: Make transfer frame size a setting.
    information_blocks = information_blocks[where_asms[0] +
                                            ASM_arr.shape[0]:(where_asms[0] + ASM_arr.shape[0] + num_bits * 8)]

    return information_blocks, BER_before_decoding, where_asms
