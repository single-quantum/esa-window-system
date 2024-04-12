# %%
# Example showing the Viterbi algorithm: https://www.youtube.com/watch?v=dKIf6mQUfnY
# Very useful video explaining what a trellis is: https://www.youtube.com/watch?v=kRIfpmiMCpU

import itertools

import numpy as np
from tqdm import tqdm
from esawindowsystem.core.utils import flatten


def euclidean_distance(a, b):
    return np.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))]))


def hamming_distance(a, b):
    """Calculate the Hamming distance between codeword `a` and `b`. """
    codewords = list(itertools.product([0, 1], repeat=len(a)))
    euclidean_distances = [euclidean_distance(a, c) for c in codewords]
    closest_codeword_idx = euclidean_distances.index(min(euclidean_distances))
    closest_codeword = codewords[closest_codeword_idx]

    b = [0 if b[i] <= 0 else 1 for i in range(len(b))]

    return sum([abs(closest_codeword[i] - b[i]) for i in range(len(a))])


def viterbi(num_output_bits, received_sequence, tr):
    time_steps = len(tr.stages)

    print('Calculating Hamming distances')

    for t in tqdm(range(time_steps)):
        r_t = received_sequence[t * num_output_bits:t * num_output_bits + num_output_bits]

        edges = flatten(list(map(lambda s: s.edges, tr.stages[t].states)))
        if t > 0:
            previous_edges = flatten(list(map(lambda s: s.edges, tr.stages[t-1].states)))

        for edge in edges:
            d = hamming_distance(r_t, edge.edge_output)
            if t == 0:
                edge.hamming_distance = d
            else:
                previous_edges_to_state = list(filter(lambda e: e.to_state == edge.from_state, previous_edges))

                # Take the edge with the smallest hamming distance
                previous_edge = min(previous_edges_to_state, key=lambda e: e.hamming_distance)
                edge.hamming_distance = previous_edge.hamming_distance + d

    num_input_bits = tr.num_input_bits

    # predict the input sequence, based on the edges with a minimum hamming distance
    input_sequence = np.zeros(int(len(received_sequence) / (num_output_bits / num_input_bits)))

    # The final stage has no edges
    edges_to_state = flatten(list(map(lambda s: s.edges, tr.stages[-2].states)))
    edge = min(edges_to_state, key=lambda e: e.hamming_distance)
    to_state = edge.from_state
    input_sequence[-num_input_bits:] = edge.edge_input

    for t in reversed(range(1, time_steps-1)):
        previous_edges = flatten(list(map(lambda s: s.edges, tr.stages[t-1].states)))
        edges_to_state = list(filter(lambda e: e.to_state == to_state, previous_edges))
        edge = min(edges_to_state, key=lambda e: e.hamming_distance)
        input_sequence[t * num_input_bits:(t + 1) * num_input_bits] = edge.edge_input

        to_state = edge.from_state

    return input_sequence
