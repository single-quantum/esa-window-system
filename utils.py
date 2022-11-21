import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from tabulate import tabulate

from encoder_functions import convolve
from ppm_parameters import (BIT_INTERLEAVE, CHANNEL_INTERLEAVE, B_interleaver,
                            M, N_interleaver, num_bins_per_symbol,
                            num_samples_per_slot)
from trellis import Edge


def print_ppm_parameters():
    var_names = [
        'M',
        'num_samples_per_slot',
        'num_bins_per_symbol',
        'BIT_INTERLEAVE',
        'CHANNEL_INTERLEAVE',
        'B_interleaver',
        'N_interleaver'
    ]

    var_values = [
        M,
        num_samples_per_slot,
        num_bins_per_symbol,
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


def bpsk(s): return tuple(1 if i else -1 for i in s)


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
            output, terminal_state = convolve([input_bit], initial_state=initial_state)
            to_state = states.index(terminal_state)
            e = Edge()
            if bpsk_encoding:
                e.set_edge(from_state, to_state, input_bit, edge_output=bpsk(output), gamma=None)
                state_edges.append(e)
            else:
                e.set_edge(from_state, to_state, input_bit, edge_output=tuple(output), gamma=None)
                state_edges.append(e)
        edges.append(state_edges)

    return edges


def AWGN(input_sequence, sigma=0.8):
    rng = default_rng()
    input_sequence = input_sequence.astype(float)
    input_sequence += rng.normal(0, sigma, size=len(input_sequence))

    return input_sequence


def flatten(list_of_lists):
    return [i for sublist in list_of_lists for i in sublist]
