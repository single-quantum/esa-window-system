import itertools
import pickle
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from tabulate import tabulate

from esawindowsystem.core.encoder_functions import convolve
from esawindowsystem.core.trellis import Edge
from esawindowsystem.ppm_parameters import (BIT_INTERLEAVE, CHANNEL_INTERLEAVE, B_interleaver,
                            M, N_interleaver, num_samples_per_slot,
                            num_slots_per_symbol)


def print_ppm_parameters():
    var_names = [
        'M',
        'num_samples_per_slot',
        'num_slots_per_symbol',
        'BIT_INTERLEAVE',
        'CHANNEL_INTERLEAVE',
        'B_interleaver',
        'N_interleaver'
    ]

    var_values = [
        M,
        num_samples_per_slot,
        num_slots_per_symbol,
        BIT_INTERLEAVE,
        CHANNEL_INTERLEAVE,
        B_interleaver,
        N_interleaver
    ]

    var_names_and_values = zip(var_names, var_values)

    print(tabulate(var_names_and_values, headers=["Variable", "Value"]))


def save_figure(plt, name, dir):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    p = Path('simulation results') / Path(date_str)
    if not p.exists():
        Path.mkdir(p)
        Path.mkdir(p / Path(dir))

    plt.savefig(p / Path(dir) / name)


def bpsk(s: list | tuple | npt.NDArray) -> tuple[int, ...]: return tuple(1 if i else -1 for i in s)


def bpsk_encoding(input_sequence):
    """Use BPSK to modulate the bit array. """
    output = np.zeros_like(input_sequence)

    for i, ri in enumerate(input_sequence):
        if ri == 0:
            output[i] = -1
        else:
            output[i] = 1

    return output


def tobits(s):
    result = []
    for c in s:
        bits = bin(ord(c))[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])
    return result


def frombits(bits):
    chars = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        chars.append(chr(int(''.join([str(bit) for bit in byte]), 2)))
    return ''.join(chars)


def generate_outer_code_edges(memory_size, bpsk_encoding=True):
    input_bits = [0, 1]
    edges = []
    states = list(itertools.product([0, 1], repeat=memory_size))

    for i, initial_state in enumerate(states):
        state_edges = []
        for input_bit in input_bits:
            from_state = i
            output, terminal_state = convolve((input_bit, ), initial_state=initial_state)
            to_state = states.index(terminal_state)
            e = Edge()
            if bpsk_encoding:
                e.set_edge(from_state, to_state, input_bit, edge_output=bpsk(output), gamma=None)
                state_edges.append(e)
            else:
                e.set_edge(from_state, to_state, input_bit, edge_output=output, gamma=None)
                state_edges.append(e)
        edges.append(state_edges)

    return edges

# Note to self: I could probably make one function for the inner/outer edge generation,
# and then put it in the Trellis constructor.


def generate_inner_encoder_edges(num_input_bits, bpsk_encoding=True):
    input_combinations: list[tuple[int, ...]] = list(itertools.product([0, 1], repeat=num_input_bits))
    edges = []
    input_bits: tuple[int]
    for initial_state in [0, 1]:
        state_edges = []
        for input_bits in input_combinations:
            # initial_state = 1
            current_state = initial_state
            output: list[int] = []

            for bit in input_bits:
                output_bit = current_state ^ bit
                output.append(output_bit)
                current_state = output_bit

            if bpsk_encoding:
                output = bpsk(output)

            e = Edge()
            e.set_edge(initial_state, current_state, input_bits, output, gamma=None)
            state_edges.append(e)

        edges.append(state_edges)

    return edges


def AWGN(input_sequence, sigma=0.8):
    """Superimpose Additive White Gaussian Noise on the input sequence. """
    rng = default_rng()
    input_sequence = input_sequence.astype(float)
    input_sequence += rng.normal(0, sigma, size=len(input_sequence))

    return input_sequence


def poisson_noise(input_sequence: npt.NDArray, ns: float, nb: float,
                  simulate_lost_symbols=False, detection_efficiency: float = 1):
    output_sequence = deepcopy(input_sequence)
    rng = default_rng()
    lost_symbols = [None for _ in range(input_sequence.shape[0])]
    if simulate_lost_symbols:
        lost_symbols = np.array(rng.random(input_sequence.shape[0]) >= detection_efficiency)

    poisson_dist_signal_slots = rng.poisson(ns+nb, size=output_sequence.shape)
    poisson_dist_noise_slots = rng.poisson(nb, size=output_sequence.shape)

    for i, row in enumerate(output_sequence):
        j = np.where(row == 1)[0][0]
        row += poisson_dist_noise_slots[i]
        if not (simulate_lost_symbols and lost_symbols[i]):
            row[j] = poisson_dist_signal_slots[i, j]

    return output_sequence


def flatten(list_of_lists: list[list]) -> list:
    """Convert a list of lists to a flat (1D) list. """
    return [i for sublist in list_of_lists for i in sublist]


def moving_average(arr: npt.NDArray[Any], n: int = 3) -> npt.NDArray[Any]:
    """Calculates the moving average of the array `a`, with a window size of `n`

    Source:
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy"""
    ret: npt.NDArray[np.float_] = np.cumsum(arr, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def check_user_settings(user_settings: dict) -> None:
    B_interleaver: int | None = user_settings.get('B_interleaver')
    if B_interleaver is None:
        raise KeyError("B_interleaver not found in `user_settings`")
    if not isinstance(B_interleaver, int):
        raise ValueError("B_interleaver should be an integer. ")

    N_interleaver: int | None = user_settings.get('N_interleaver')
    if N_interleaver is None:
        raise KeyError("N_interleaver not found in `user_settings`")
    if not isinstance(N_interleaver, int):
        raise ValueError("N_interleaver should be an integer. ")


def get_BER_before_decoding(bit_sequence_file_path, received_bits, sent_bit_sequence=None):
    if sent_bit_sequence is None:
        with open(bit_sequence_file_path, 'rb') as f:
            sent_bits = pickle.load(f)
    else:
        sent_bits = sent_bit_sequence

    BER_before_decoding = np.sum([abs(x - y) for x, y in zip(received_bits, sent_bits)])

    return BER_before_decoding


def ppm_symbols_to_bit_array(received_symbols: npt.ArrayLike, m: int = 4) -> npt.NDArray[np.int_]:
    """Map PPM symbols back to bit array. """
    received_symbols = np.array(received_symbols)
    reshaped_ppm_symbols = received_symbols.astype(np.uint8).reshape(received_symbols.shape[0], 1)
    bits_array = np.unpackbits(reshaped_ppm_symbols, axis=1).astype(int)
    received_sequence: npt.NDArray[np.int_] = bits_array[:, -m:].reshape(-1)

    return received_sequence
