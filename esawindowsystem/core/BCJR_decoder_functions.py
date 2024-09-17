import itertools
from copy import deepcopy
from fractions import Fraction
from itertools import chain
from math import exp, prod

import numpy as np
import numpy.typing as npt
from numpy import dot
from tqdm import tqdm

# from esawindowsystem.core.max_star import max_star, max_star_recursive
from esawindowsystem.core.BCJR_decoder_utils import (max_star_lru,
                                                     max_star_recursive)
from esawindowsystem.core.encoder_functions import (BitArray, bit_deinterleave,
                                                    bit_interleave,
                                                    channel_deinterleave,
                                                    get_CRC, get_csm,
                                                    get_remap_indices,
                                                    randomize, unpuncture)
from esawindowsystem.core.scppm_encoder import puncture
from esawindowsystem.core.trellis import Edge, Trellis
from esawindowsystem.core.utils import (flatten, generate_inner_encoder_edges,
                                        generate_outer_code_edges,
                                        poisson_noise)


def gamma_awgn(r, v, Es, N0): return exp(Es / N0 * 2 * dot(r, v))
def log_gamma(r, v, Es, N0): return Es / N0 * 2 * dot(r, v)


def calculate_alphas(trellis: Trellis, log_bcjr: bool = True, verbose: bool = False) -> None:
    """ Calculate the alpha for each state in the trellis.

    Alpha is a likelihood that has a backward recursion relation to previous states.
    It says something about the likelihood of being in that state,
    given the history of the received sequence up to that state.
    Alpha is calculated by taking each edge that is connected to the previous state and weighing it with gamma.

    `log_bcjr` determines whether or not to take the log of alpha, which turns a multiplication into a sum.
    For more information, see Moison and Hamkins (2005), section 3.C."""
    if verbose:
        print('Calculating alphas')

    # Encoder is initiated in the all zeros state, so only alpha[0, 0] is non-zero for the first column
    if log_bcjr:
        trellis.stages[0].states[0].alpha = 0
        trellis.stages[0].states[1].alpha = -np.inf
        trellis.stages[0].states[2].alpha = -np.inf
        trellis.stages[0].states[3].alpha = -np.inf
    else:
        trellis.stages[0].states[0].alpha = 1

    time_steps: int = len(trellis.stages)

    for i in tqdm(range(1, time_steps), leave=False):
        for state in trellis.stages[i].states:
            alpha_ji: list[float] = []

            # Get a list of all edges in the previous stage
            previous_states = trellis.stages[i - 1].states
            previous_edges: list[Edge] = flatten([s.edges for s in previous_states])

            # Find all the edges that go to the current state
            edges: list[Edge] = [e for e in previous_edges if e.to_state == state.label]

            for edge in edges:
                previous_state = previous_states[edge.from_state]
                if log_bcjr:
                    alpha_ji.append(previous_state.alpha + edge.gamma)
                else:
                    alpha_ji.append(previous_state.alpha * edge.gamma)

            if log_bcjr and len(alpha_ji) > 1:
                state.alpha = max_star_recursive(alpha_ji)
            elif log_bcjr and alpha_ji:
                state.alpha = alpha_ji[0]
            else:
                state.alpha = sum(alpha_ji)

        # Normalize each alpha to prevent overflow and improve performance.
        if not log_bcjr:
            sum_of_alphas = sum([s.alpha for s in trellis.stages[i].states])
            for state in trellis.stages[i].states:
                state.alpha = state.alpha / sum_of_alphas


def calculate_alpha_inner_SISO(trellis: Trellis, gamma_primes, log_bcjr=True):
    """ Calculate the alpha for each state in the trellis.

    Alpha is a likelihood that has a backward recursion relation to previous states.
    It says something about the likelihood of being in that state,
    given the history of the received sequence up to that state.
    Alpha is calculated by taking each edge that is connected to the previous state and weighing it with gamma. """
    # print('Calculating alphas')

    # Encoder is initiated in the all zeros state, so only alpha[0, 0] is non-zero for the first column
    if log_bcjr:
        trellis.stages[0].states[0].alpha = 0
        trellis.stages[0].states[1].alpha = -np.inf
    else:
        trellis.stages[0].states[0].alpha = 1

    time_steps = len(trellis.stages)

    for k in range(1, time_steps):
        stage = trellis.stages[k]
        previous_stage = trellis.stages[k - 1]
        previous_states = previous_stage.states

        for state in stage.states:
            a0 = previous_states[0].alpha + gamma_primes[0, state.label, k - 1]
            if k == 1:
                state.alpha = max_star_lru(a0, -np.inf)
            else:
                a1 = previous_states[1].alpha + gamma_primes[1, state.label, k - 1]
                state.alpha = max_star_lru(a0, a1)


def calculate_beta_inner_SISO(trellis: Trellis, gamma_primes, log_bcjr: bool = True):
    """ Calculate the beta for each state in the trellis.

    Beta is a likelihood that has a forward recursion relation to next states.
    It says something about the likelihood of being in that state,
    given the future of the received sequence up to that state.
    Beta is calculated by taking each edge that is connected to the given state and weighing it with gamma. """
    # print('Calculating betas')

    # Encoder is initiated in the all zeros state, so only alpha[0, 0] is non-zero for the first column
    if log_bcjr:
        trellis.stages[-1].states[0].beta = 0
        trellis.stages[-1].states[1].beta = 0
    else:
        trellis.stages[-1].states[0].beta = 1 / 2
        trellis.stages[-1].states[1].beta = 1 / 2

    time_steps = len(trellis.stages)

    for k in reversed(range(0, time_steps - 1)):
        stage = trellis.stages[k]
        next_stage = trellis.stages[k + 1]
        next_states = next_stage.states

        for state in stage.states:
            b0 = next_states[0].beta + gamma_primes[state.label, 0, k]
            b1 = next_states[1].beta + gamma_primes[state.label, 1, k]
            state.beta = max_star_lru(b0, b1)


def calculate_betas(trellis: Trellis, log_bcjr: bool = True, verbose: bool = False) -> None:
    if verbose:
        print('Calculating betas')
    # Betas are also likelihoods, but unlike alpha, they have a forward recursion relation.
    # Same here, but now zero terminated
    if log_bcjr:
        trellis.stages[-1].states[0].beta = 0
        trellis.stages[-1].states[1].beta = -np.inf
        trellis.stages[-1].states[2].beta = -np.inf
        trellis.stages[-1].states[3].beta = -np.inf
    else:
        trellis.stages[-1].states[0].beta = 1

    time_steps = len(trellis.stages) - 1

    with tqdm(total=time_steps, leave=False) as pbar:
        for i in reversed(range(0, time_steps)):
            stage = trellis.stages[i]
            next_stage = trellis.stages[i + 1]
            for j, state in enumerate(stage.states):
                beta_ji = []

                for edge in state.edges:
                    if log_bcjr:
                        beta_ji.append(next_stage.states[edge.to_state].beta + edge.gamma)
                    else:
                        beta_ji.append(next_stage.states[edge.to_state].beta * edge.gamma)

                if log_bcjr and len(beta_ji) > 1:
                    state.beta = max_star_recursive(beta_ji)
                elif log_bcjr and beta_ji:
                    state.beta = beta_ji[0]
                else:
                    state.beta = sum(beta_ji)

            # Normalize each column to prevent overflow and improve performance.
            if not log_bcjr:
                sum_of_betas = sum([s.beta for s in trellis.stages[i].states])
                for state in trellis.stages[i].states:
                    state.beta = state.beta / sum_of_betas
            pbar.update(1)


def calculate_gammas(trellis: Trellis, received_sequence, num_output_bits, Es, N0, log_bcjr=True, verbose=False):
    if verbose:
        print('Calculating gammas')

    # Gamma values are a certain weight coupled to each edge.
    for k, stage in tqdm(enumerate(trellis.stages[:-1]), leave=False):
        for state in stage.states:
            for edge in state.edges:
                # received_codeword = received_sequence[k, :]
                r = received_sequence[k * num_output_bits:(k + 1) * num_output_bits]
                edge_output = edge.edge_output

                if log_bcjr:
                    g = log_gamma(r, edge_output, Es, N0)
                else:
                    g = gamma_awgn(r, edge_output, Es, N0)

                edge.gamma = g


def calculate_gamma_inner_SISO(trellis: Trellis, symbol_bit_LLRs, channel_log_likelihoods):
    for k, stage in enumerate(trellis.stages[:-1]):
        for state in stage.states:
            for edge in state.edges:
                edge.gamma = pi_ak(edge.edge_input, symbol_bit_LLRs[k]) + \
                    channel_log_likelihoods[k, edge.edge_output_label]


def calculate_gamma_primes(trellis: Trellis):
    gamma_prime = np.zeros((2, 2, len(trellis.stages[:-1])))
    combinations = list(itertools.product([0, 1], repeat=2))
    for k, stage in enumerate(trellis.stages[:-1]):
        for i, j in combinations:
            gammas = [e.gamma for e in stage.states[i].edges if (e.from_state == i and e.to_state == j)]
            if not gammas:
                continue
            gamma_prime[i, j, k] = max_star_recursive(gammas)

    return gamma_prime


def calculate_LLRs(trellis: Trellis, log_bcjr=True, verbose=False) -> npt.NDArray[np.float64]:
    """ Calculate the Log likelihoods given a set of alphas, gammas and betas.

    The Log-likelihood Ratio (LLR) is the ratio between two a posteriori probabilies.
    The numerator is the probability of the message bit being 0, given the received sequence, while the denominator
    is the probability of the message bit being 1, given the received sequence.
    To calculate this, you need to sum over all states that had a 0 as input for the numerator,
    While summing over all the states that had a 1 as input for the denominator.
    """
    if verbose:
        print('Calculate log likelihoods')
    time_steps: int = len(trellis.stages) - 1
    LLR = np.zeros(time_steps, dtype=float)

    # The last stage has no edges
    for t, stage in enumerate(trellis.stages[:-1]):
        numerator = []
        denomenator = []
        for state in stage.states:
            for edge in state.edges:
                to_state = edge.to_state

                likelihood_terms = [state.alpha, edge.gamma, trellis.stages[t + 1].states[to_state].beta]
                likelihood = sum(likelihood_terms) if log_bcjr else prod(likelihood_terms)

                if edge.edge_input == 1:
                    numerator.append(likelihood)
                else:
                    denomenator.append(likelihood)

        if not log_bcjr:
            LLR[t] = np.log(sum(numerator) / sum(denomenator))
            continue

        # When leaving from the first state in the first timestep, there's only two transitions possible
        if len(numerator) == 1 and len(denomenator) == 1:
            LLR[t] = numerator[0] - denomenator[0]
        # Termination phase
        elif len(numerator) == 0 and len(denomenator) >= 1:
            LLR[t] = -np.inf
        else:
            LLR[t] = max_star_recursive(numerator) - max_star_recursive(denomenator)

    return LLR


def calculate_inner_SISO_LLRs(trellis: Trellis, symbol_bit_LLRs):
    """ Calculate the Log likelihoods given a set of alphas, gammas and betas.

    The Log-likelihood Ratio (LLR) is the ratio between two a posteriori probabilies.
    The numerator is the probability of the message bit being 0, given the received sequence, while the denominator
    is the probability of the message bit being 1, given the received sequence.
    To calculate this, you need to sum over all states that had a 0 as input for the numerator,
    While summing over all the states that had a 1 as input for the denominator.

    The `bit_log_likelihoods` variable is a vector of vectors that contains the bit log likelihoods
    for each k-th PPM symbol.
    """
    # print('Calculate log likelihoods')
    time_steps = len(trellis.stages) - 1
    m = symbol_bit_LLRs.shape[1]
    LLRs = np.zeros((time_steps, m))

    for k, stage in enumerate(trellis.stages[:-1]):
        next_states = trellis.stages[k + 1].states
        for state in stage.states:
            for edge in state.edges:
                next_state = next_states[edge.to_state]
                edge.lmbda = state.alpha + edge.gamma + next_state.beta

    for k, stage in enumerate(trellis.stages[:-1]):
        edges = flatten(list(map(lambda s: s.edges, stage.states)))

        for i in range(trellis.num_input_bits):
            # Get all lambdas for edges that have 0 / 1 as input
            zero_edges_lmbdas = [e.lmbda for e in edges if e.edge_input[i] == 0]
            ones_edges_lmbdas = [e.lmbda for e in edges if e.edge_input[i] == 1]

            LLRs[k, i] = max_star_recursive(zero_edges_lmbdas) - \
                max_star_recursive(ones_edges_lmbdas) - symbol_bit_LLRs[k, i]

    return LLRs


def calculate_outer_SISO_LLRs(trellis: Trellis, symbol_bit_LLRs, log_bcjr=True):
    """ Calculate the Log likelihoods given a set of alphas, gammas and betas.

    The Log-likelihood Ratio (LLR) is the ratio between two a posteriori probabilies.
    The numerator is the probability of the message bit being 0, given the received sequence, while the denominator
    is the probability of the message bit being 1, given the received sequence.
    To calculate this, you need to sum over all states that had a 0 as input for the numerator,
    While summing over all the states that had a 1 as input for the denominator.

    The `bit_log_likelihoods` variable is a vector of vectors that contains the bit log likelihoods
    for each k-th PPM symbol.
    """
    # print('Calculate log likelihoods')
    time_steps = len(trellis.stages) - 1
    symbol_bit_LLRs = symbol_bit_LLRs.reshape((-1, 3))
    p_xk_O = np.zeros((time_steps, trellis.num_output_bits))
    p_uk_O = np.zeros(time_steps)

    for k, stage in enumerate(trellis.stages[:-1]):
        next_states = trellis.stages[k + 1].states
        for state in stage.states:
            for edge in state.edges:
                next_state = next_states[edge.to_state]
                edge.lmbda = state.alpha + edge.gamma + next_state.beta

    for k, stage in enumerate(trellis.stages[:-1]):
        edges: list[Edge] = flatten(list(map(lambda s: s.edges, stage.states)))

        for i in range(trellis.num_output_bits):
            # First, select all edges with 0 (or 1) at the i-th output bit, then of those edges, take the lambda value.
            zero_edges_lmbdas = [e.lmbda for e in edges if e.edge_output[i] == 0]
            ones_edges_lmbdas = [e.lmbda for e in edges if e.edge_output[i] == 1]

            if len(zero_edges_lmbdas) == 1 and len(ones_edges_lmbdas) == 1:
                p_xk_O[k, i] = max_star_lru(zero_edges_lmbdas[0], -np.inf) - \
                    max_star_lru(ones_edges_lmbdas[0], -np.inf) - symbol_bit_LLRs[k, i]
                continue
            if len(ones_edges_lmbdas) == 0 and len(zero_edges_lmbdas) != 0:
                p_xk_O[k, i] = max_star_recursive(zero_edges_lmbdas) - symbol_bit_LLRs[k, i]
                continue

            p_xk_O[k, i] = max_star_recursive(zero_edges_lmbdas) - \
                max_star_recursive(ones_edges_lmbdas) - symbol_bit_LLRs[k, i]

        zero_edges_lmbdas = [e.lmbda for e in edges if e.edge_input == 0]
        ones_edges_lmbdas = [e.lmbda for e in edges if e.edge_input == 1]

        if len(zero_edges_lmbdas) == 1 and len(ones_edges_lmbdas) == 1:
            p_uk_O[k] = max_star_lru(zero_edges_lmbdas[0], -np.inf) - max_star_lru(ones_edges_lmbdas[0], -np.inf)
        elif len(ones_edges_lmbdas) == 0 and len(zero_edges_lmbdas) != 0:
            p_uk_O[k] = max_star_recursive(zero_edges_lmbdas)
        else:
            p_uk_O[k] = max_star_recursive(zero_edges_lmbdas) - max_star_recursive(ones_edges_lmbdas)

    return p_xk_O, p_uk_O


def predict(trellis: Trellis, received_sequence, LOG_BCJR=True, Es=10, N0=1, verbose=False):
    """Use the BCJR algorithm to predict the sent message, based on the received sequence. """
    time_steps = len(received_sequence)

    # Calculate alphas, betas, gammas and LLRs
    calculate_gammas(trellis, received_sequence, trellis.num_output_bits, Es, N0, log_bcjr=LOG_BCJR)
    calculate_alphas(trellis, log_bcjr=LOG_BCJR)
    calculate_betas(trellis, log_bcjr=LOG_BCJR)
    LLRs = calculate_LLRs(trellis, log_bcjr=LOG_BCJR)

    if verbose:
        print('Message decoded')
    u_hat = np.array([1 if llr >= 0 else 0 for llr in LLRs])

    return u_hat


def predict_inner_SISO(trellis: Trellis, channel_log_likelihoods, time_steps, m, symbol_bit_LLRs=None):
    if symbol_bit_LLRs is None:
        symbol_bit_LLRs = np.zeros((time_steps, m))

    # Calculate alphas, betas, gammas and LLRs
    calculate_gamma_inner_SISO(trellis, symbol_bit_LLRs, channel_log_likelihoods)
    gamma_primes = calculate_gamma_primes(trellis)

    calculate_alpha_inner_SISO(trellis, gamma_primes)
    calculate_beta_inner_SISO(trellis, gamma_primes)
    LLRs = calculate_inner_SISO_LLRs(trellis, symbol_bit_LLRs)

    return LLRs


def ppm_symbols_to_bit_array(received_symbols: npt.ArrayLike, m: int = 4) -> npt.NDArray[np.int_]:
    """Map PPM symbols back to bit array. """
    received_symbols = np.array(received_symbols)
    reshaped_ppm_symbols = received_symbols.astype(np.uint8).reshape(received_symbols.shape[0], 1)
    bits_array = np.unpackbits(reshaped_ppm_symbols, axis=1).astype(int)
    received_sequence: npt.NDArray[np.int_] = bits_array[:, -m:].reshape(-1)

    return received_sequence


def pi_ck(
        input_sequence: npt.NDArray[np.float64],
        ns: float,
        nb: float) -> npt.NDArray[np.float64]:
    """Calculate symbol log likelihood, based on likelihoods from the channel (Poisson statistics).

    This formula is given in Moision on page 12, below formula 13.
    """
    output_sequence = deepcopy(input_sequence)
    output_sequence = output_sequence * np.log(1 + ns / nb) / np.log(ns / nb)

    return output_sequence


def pi_ak(PPM_symbol_vector, bit_LLRs):
    """Calculate symbol log likelihood, based on bit likelihoods from outer SISO

    This formula is given in Moision on page 9, below formula 10. """
    return sum([0.5 * (-1)**PPM_symbol_vector[i] * bit_LLRs[i] for i in range(len(bit_LLRs))])


edge_gamma_partial: dict[tuple[int, int, int], npt.NDArray[np.int_]] = {}
for idx, _ in np.ndenumerate(np.zeros((2, 2, 2))):
    edge_gamma_partial[(idx[0], idx[1], idx[2])] = (-1)**np.array([idx[0], idx[1], idx[2]])


def set_outer_code_gammas(trellis: Trellis, symbol_log_likelihoods: npt.NDArray[np.float64]):
    symbol_log_likelihoods = symbol_log_likelihoods.reshape((-1, 3))

    for k, stage in enumerate(trellis.stages[:-1]):
        symbol_llrs = symbol_log_likelihoods[k, :]
        for si, state in enumerate(stage.states):
            for ei, edge in enumerate(state.edges):
                e: npt.NDArray[np.int_] = edge.edge_output
                gamma_partial: npt.NDArray[np.int_] = edge_gamma_partial[(e[0], e[1], e[2])]
                edge.gamma = np.sum(0.5 * gamma_partial * symbol_llrs)


def set_outer_code_gammas_arr(
        trellis: Trellis,
        edge_outputs: npt.NDArray[np.int_],
        symbol_log_likelihoods: npt.NDArray[np.float64]) -> None:
    """Set outer code gammas based on edge output array. """

    symbol_log_likelihoods = symbol_log_likelihoods.reshape((-1, 3))
    edge_gammas: npt.NDArray[np.float64] = np.sum(0.5 * ((-1)**edge_outputs) * symbol_log_likelihoods, axis=3)
    return edge_gammas


def get_edge_output_array(trellis: Trellis) -> npt.NDArray[np.int_]:
    num_stages = len(trellis.stages) - 1
    stage = trellis.stages[0]
    num_states = len(stage.states)
    num_edges = len(stage.states[0].edges)

    edge_outputs = np.empty(
        (num_stages, num_states, num_edges, len(stage.states[0].edges[0].edge_output))
    )
    edge_outputs[:] = np.nan

    for i, stage in enumerate(trellis.stages[:num_stages]):
        for j, state in enumerate(stage.states):
            for k, edge in enumerate(state.edges):
                edge_outputs[i, j, k, :] = edge.edge_output

    edge_outputs = np.transpose(edge_outputs, (2, 1, 0, 3))
    return edge_outputs


def predict_iteratively(slot_mapped_sequence: npt.NDArray[np.int_], M: int, code_rate: Fraction, max_num_iterations: int = 20,
                        ns: float = 3, nb: float = 0.1, ber_stop_threshold: float = 1E-7, **kwargs):
    # Initialize outer trellis edges
    memory_size_outer = 2
    num_output_bits_outer = 3
    num_input_bits_outer = 1
    outer_edges = generate_outer_code_edges(memory_size_outer, bpsk_encoding=False)

    # Initialize inner trellis edges
    m = int(np.log2(M))
    memory_size = 1
    num_output_bits = m
    num_input_bits = m
    inner_edges = generate_inner_encoder_edges(m, bpsk_encoding=False)

    information_block_sizes: dict[Fraction, int] = {
        Fraction(1, 3): 5040,
        Fraction(1, 2): 7560,
        Fraction(2, 3): 10080
    }

    num_bits_per_slice = information_block_sizes[code_rate]
    num_symbols_per_slice = int(num_bits_per_slice * 1 / code_rate / m)
    num_slices = int((slot_mapped_sequence.shape[0] * m * code_rate) / num_bits_per_slice)

    num_events_per_slot = kwargs.get('num_events_per_slot')
    # num_events_per_slot = None
    if num_events_per_slot is not None:
        N_interleaver = 2
        B_interleaver = int(15120 / m / N_interleaver)
        CSM = get_csm(M)
        reshaped_num_events = num_events_per_slot.reshape(-1,
                                                          num_symbols_per_slice + len(CSM), int(5 / 4 * M))[:, len(CSM):, :M]

        # "flatten" it again
        reshaped_num_events = reshaped_num_events.reshape(-1, M)
        reshaped_num_events = channel_deinterleave(reshaped_num_events, B_interleaver, N_interleaver)

        remapped_indices = get_remap_indices(reshaped_num_events, B_interleaver, N_interleaver)
        new_remapped_indices = np.zeros(reshaped_num_events.shape[0], dtype=int)
        for i, idx in enumerate(remapped_indices):
            if idx >= reshaped_num_events.shape[0]:
                continue
            else:
                new_remapped_indices[i] = idx

        # reshaped_num_events = reshaped_num_events[new_remapped_indices]
        num_zeros_interleaver = 2 * B_interleaver * N_interleaver * (N_interleaver - 1)
        # reshaped_num_events = num_events_per_slot.reshape(num_slices, num_symbols_per_slice+len(CSM), int(5/4*M))[:, len(CSM):, :M]
        channel_likelihoods = reshaped_num_events[:-num_zeros_interleaver].astype(int)

    else:
        channel_likelihoods = poisson_noise(
            slot_mapped_sequence[:, :M],
            ns=ns,
            nb=nb,
            simulate_lost_symbols=kwargs.get('simulate_lost_symbols', False),
            detection_efficiency=kwargs.get('detection_efficiency', 1)
        )

    decoded_message = []
    decoded_message_array = np.zeros((max_num_iterations, num_slices, num_bits_per_slice))

    sent_bit_sequence: BitArray | None = kwargs.get('sent_bit_sequence_no_csm')

    bit_error_ratios = np.zeros((max_num_iterations, num_slices))

    inner_trellis = Trellis(memory_size, num_output_bits, num_symbols_per_slice, inner_edges, num_input_bits)
    inner_trellis.set_edges(inner_edges, zero_terminated=False)

    outer_trellis = Trellis(memory_size_outer, num_output_bits_outer,
                            num_bits_per_slice, outer_edges, num_input_bits_outer)
    outer_trellis.set_edges(outer_edges)

    for i in range(num_slices):
        print(f'Decoding slice {i+1}/{num_slices}')
        # Generate a vector with a poisson distributed number of photons per slot
        # Calculate the corresponding log likelihood
        channel_log_likelihoods = pi_ck(
            channel_likelihoods[i * num_symbols_per_slice:(i + 1) * num_symbols_per_slice], ns, nb)

        time_steps_inner = num_symbols_per_slice

        symbol_bit_LLRs = None
        u_hat = []

        for iteration in range(max_num_iterations):
            print(f'Iteration {iteration+1}/{max_num_iterations}')
            p_ak_O = predict_inner_SISO(inner_trellis, channel_log_likelihoods,
                                        time_steps_inner, m, symbol_bit_LLRs=symbol_bit_LLRs)
            p_xk_I = bit_deinterleave(p_ak_O.flatten(), dtype=float)

            p_xk_I = unpuncture(p_xk_I, code_rate, dtype=float)

            set_outer_code_gammas(outer_trellis, p_xk_I)
            calculate_alphas(outer_trellis)
            calculate_betas(outer_trellis)
            p_xk_O, LLRs_u = calculate_outer_SISO_LLRs(outer_trellis, p_xk_I)
            p_xk_O = puncture(np.array([p_xk_O.flatten()]), code_rate, dtype=float)
            p_ak_I = bit_interleave(p_xk_O.flatten(), dtype=float)

            symbol_bit_LLRs = deepcopy(p_ak_I.reshape(-1, m))

            u_hat = [0 if llr > 0 else 1 for llr in LLRs_u]

            # Derandomize
            u_hat = randomize(np.array(u_hat, dtype=int))

            ber: float = 1
            sent_bits_codeword: BitArray
            if sent_bit_sequence is not None:
                sent_bits_codeword = sent_bit_sequence[
                    i * num_bits_per_slice - 2 * i:(i + 1) * num_bits_per_slice - 2 * (i + 1)
                ]
                ber = np.sum(
                    [abs(x - y) for x, y in zip(u_hat, sent_bits_codeword)]
                ) / num_bits_per_slice
                print(
                    f"iteration = {iteration+1} ber: {ber:.3e} \t min likelihood: " +
                    f"{np.min(LLRs_u):.2f} \t max likelihood: {np.max(LLRs_u):.2f}")

                bit_error_ratios[iteration, i] = ber

            decoded_message_array[iteration, i, :] = u_hat

            if ber < ber_stop_threshold:
                break

        decoded_message.append(u_hat)

    # Flatten and cast to numpy array
    decoded_message = np.array([bit for sublist in decoded_message for bit in sublist], dtype=int)

    return decoded_message, decoded_message_array, bit_error_ratios
