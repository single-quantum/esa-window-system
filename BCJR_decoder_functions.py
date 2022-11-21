import itertools
from itertools import chain
from math import exp

import numpy as np
import numpy.typing as npt
from numpy import dot
from tqdm import tqdm

from utils import flatten


def gamma_awgn(r, v, Es, N0): return exp(Es / N0 * 2 * dot(r, v))
def log_gamma(r, v, Es, N0): return Es / N0 * 2 * dot(r, v)


keys = list(itertools.product(range(6), repeat=2))
max_log_lookup = {}
for key in keys:
    max_log_lookup[key] = np.log(1 + np.exp(-abs(key[0] - key[1])))


def max_star(a, b):
    if abs(a) > 5 or abs(b) > 5 or abs(a - b) > 5:
        return max(a, b)
    elif a == -np.inf or b == -np.inf:
        return max(a, b) + np.log(2)
    elif a < 0 or b < 0:
        return max_log_lookup[(int(abs(a - b)), 0)]
    else:
        return max(a, b) + max_log_lookup[(int(round(a)), int(round(b)))]


def max_star_recursive(arr):
    result = max_star(arr[0], arr[1])
    i = 2
    while i < len(arr):
        result = max_star(result, arr[i])
        i += 1

    return result


def calculate_alphas(trellis, alpha, log_bcjr=True, verbose=False):
    """ Calculate the alpha for each state in the trellis.

    Alpha is a likelihood that has a backward recursion relation to previous states.
    It says something about the likelihood of being in that state, given the history of the received sequence up to that state.
    Alpha is calculated by taking each edge that is connected to the previous state and weighing it with gamma. """
    if verbose:
        print('Calculating alphas')

    # Encoder is initiated in the all zeros state, so only alpha[0, 0] is non-zero for the first column
    if log_bcjr:
        trellis.stages[0].states[0].alpha = 0
    else:
        trellis.stages[0].states[0].alpha = 1

    time_steps = len(trellis.stages)

    for i in tqdm(range(1, time_steps), leave=False):
        for state in trellis.stages[i].states:
            alpha_ji = []

            # Get a list of all edges in the previous stage
            previous_states = trellis.stages[i - 1].states
            previous_edges = list(chain(*map(lambda s: s.edges, previous_states)))

            # Find all the edges that go to the current state
            edges = list(filter(lambda e: e.to_state == state.label, previous_edges))

            for edge in edges:
                previous_state = previous_states[edge.from_state]
                if log_bcjr:
                    alpha_ji.append(previous_state.alpha + edge.gamma)
                else:
                    alpha_ji.append(edge.alpha * edge.gamma)

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

    return alpha


def calculate_alpha_inner_SISO(trellis, gamma_primes, log_bcjr=True):
    """ Calculate the alpha for each state in the trellis.

    Alpha is a likelihood that has a backward recursion relation to previous states.
    It says something about the likelihood of being in that state, given the history of the received sequence up to that state.
    Alpha is calculated by taking each edge that is connected to the previous state and weighing it with gamma. """
    # print('Calculating alphas')

    # Encoder is initiated in the all zeros state, so only alpha[0, 0] is non-zero for the first column
    if log_bcjr:
        trellis.stages[0].states[0].alpha = 0
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
                state.alpha = max_star_recursive([a0, -np.infty])
            else:
                a1 = previous_states[1].alpha + gamma_primes[1, state.label, k - 1]
                state.alpha = max_star_recursive([a0, a1])


def calculate_beta_inner_SISO(trellis, gamma_primes, log_bcjr=True):
    """ Calculate the beta for each state in the trellis.

    Beta is a likelihood that has a forward recursion relation to next states.
    It says something about the likelihood of being in that state, given the future of the received sequence up to that state.
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
            # if k == 1:
            #     state.alpha = max_star_recursive([b0, -np.infty])
            # else:
            b1 = next_states[1].beta + gamma_primes[state.label, 1, k]
            state.beta = max_star_recursive([b0, b1])


def calculate_betas(trellis, beta, log_bcjr=True, verbose=False):
    if verbose:
        print('Calculating betas')
    # Betas are also likelihoods, but unlike alpha, they have a forward recursion relation.
    # Same here, but now zero terminated
    if log_bcjr:
        trellis.stages[-1].states[0].beta = 0
    else:
        trellis.stages[-1].states[0].beta = 1

    num_states = len(trellis.stages[0].states)
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

    return beta


def pi_k(PPM_symbol_vector, bit_LLR):
    return np.sum([0.5 * (-1)**PPM_symbol_vector[i] * bit_LLR[i] for i in range(len(bit_LLR))])


def calculate_gammas(trellis, received_sequence, num_output_bits, Es, N0, log_bcjr=True, verbose=False):
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
                    # g = pi_k(edge.edge_output, received_codeword)
                else:
                    g = gamma_awgn(r, edge_output, Es, N0)

                edge.gamma = g


def calculate_gamma_inner_SISO(trellis, symbol_bit_LLRs, symbol_channel_LLRs):
    for k, stage in enumerate(trellis.stages[:-1]):
        for state in stage.states:
            for edge in state.edges:
                edge.gamma = pi_k(edge.edge_input, symbol_bit_LLRs[k]) + pi_k(edge.edge_output, symbol_channel_LLRs[k])
                # print(edge.from_state, edge.to_state, symbol_bit_LLRs[k], edge.edge_output, edge.gamma)


def calculate_gamma_primes(trellis):
    gamma_prime = np.zeros((2, 2, len(trellis.stages[:-1])))
    for k, stage in enumerate(trellis.stages[:-1]):
        combinations = list(itertools.product([0, 1], repeat=2))
        for state in stage.states:
            for i, j in combinations:
                edges = filter(lambda e: e.from_state == i and e.to_state == j, state.edges)
                gammas = list(map(lambda e: e.gamma, edges))
                if not gammas:
                    continue
                gamma_prime[i, j, k] = max_star_recursive(gammas)

    return gamma_prime


def calculate_LLRs(trellis, alpha, beta, log_bcjr=True, verbose=False):
    """ Calculate the Log likelihoods given a set of alphas, gammas and betas.

    The Log-likelihood Ratio (LLR) is the ratio between two a posteriori probabilies.
    The numerator is the probability of the message bit being 0, given the received sequence, while the denominator
    is the probability of the message bit being 1, given the received sequence.
    To calculate this, you need to sum over all states that had a 0 as input for the numerator,
    While summing over all the states that had a 1 as input for the denominator.
    """
    if verbose:
        print('Calculate log likelihoods')
    time_steps = len(trellis.stages) - 1
    LLR = np.zeros(time_steps)

    # The last stage has no edges
    for t, stage in enumerate(trellis.stages[:-1]):
        numerator = 0
        denomenator = 0

        a = []
        b = []
        for state in stage.states:
            for edge in state.edges:
                from_state = edge.from_state
                to_state = edge.to_state

                if log_bcjr:
                    if edge.edge_input == 1:
                        a.append(state.alpha + edge.gamma + trellis.stages[t + 1].states[to_state].beta)
                    else:
                        b.append(state.alpha + edge.gamma + trellis.stages[t + 1].states[to_state].beta)
                else:
                    if edge.edge_input == 1:
                        numerator += alpha[from_state, t] * edge.gamma * beta[to_state, t + 1]
                    else:
                        denomenator += alpha[from_state, t] * edge.gamma * beta[to_state, t + 1]

        if not log_bcjr:
            LLR[t] = np.log(numerator / denomenator)
            return

        # When leaving from the first state in the first timestep, there's only two transitions possible
        if len(a) == 1 and len(b) == 1:
            LLR[t] = a[0] - b[0]
        # Termination phase
        elif len(a) == 0 and len(b) >= 1:
            LLR[t] = -np.inf
        else:
            LLR[t] = max_star_recursive(a) - max_star_recursive(b)

    return LLR


def calculate_inner_SISO_LLRs(trellis, received_symbols, symbol_bit_LLRs, log_bcjr=True):
    """ Calculate the Log likelihoods given a set of alphas, gammas and betas.

    The Log-likelihood Ratio (LLR) is the ratio between two a posteriori probabilies.
    The numerator is the probability of the message bit being 0, given the received sequence, while the denominator
    is the probability of the message bit being 1, given the received sequence.
    To calculate this, you need to sum over all states that had a 0 as input for the numerator,
    While summing over all the states that had a 1 as input for the denominator.

    The `bit_log_likelihoods` variable is a vector of vectors that contains the bit log likelihoods for each k-th PPM symbol.
    """
    # print('Calculate log likelihoods')
    time_steps = len(trellis.stages) - 1
    LLRs = np.zeros((time_steps, trellis.num_input_bits))

    for k, stage in enumerate(trellis.stages[:-1]):
        next_states = trellis.stages[k + 1].states
        for state in stage.states:
            for edge in state.edges:
                next_state = next_states[edge.to_state]
                edge.lmbda = state.alpha + edge.gamma + next_state.beta

    for k, stage in enumerate(trellis.stages[:-1]):
        edges = flatten(list(map(lambda s: s.edges, stage.states)))

        for i in range(trellis.num_input_bits):
            input_bit = received_symbols[k, i]
            if input_bit == 0:
                zero_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input[i] == input_bit, edges)))
                ones_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input[i] != input_bit, edges)))
            else:
                ones_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input[i] == input_bit, edges)))
                zero_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input[i] != input_bit, edges)))

            LLRs[k, i] = max_star_recursive(zero_edges) - max_star_recursive(ones_edges) - symbol_bit_LLRs[k, i]

    return LLRs


def calculate_outer_SISO_LLRs(trellis, received_symbols, symbol_bit_LLRs, log_bcjr=True):
    """ Calculate the Log likelihoods given a set of alphas, gammas and betas.

    The Log-likelihood Ratio (LLR) is the ratio between two a posteriori probabilies.
    The numerator is the probability of the message bit being 0, given the received sequence, while the denominator
    is the probability of the message bit being 1, given the received sequence.
    To calculate this, you need to sum over all states that had a 0 as input for the numerator,
    While summing over all the states that had a 1 as input for the denominator.

    The `bit_log_likelihoods` variable is a vector of vectors that contains the bit log likelihoods for each k-th PPM symbol.
    """
    # print('Calculate log likelihoods')
    time_steps = len(trellis.stages) - 1
    LLRs = np.zeros((time_steps, trellis.num_output_bits))

    for k, stage in enumerate(trellis.stages[:-1]):
        next_states = trellis.stages[k + 1].states
        for state in stage.states:
            for edge in state.edges:
                next_state = next_states[edge.to_state]
                edge.lmbda = state.alpha + edge.gamma + next_state.beta

    for k, stage in enumerate(trellis.stages[:-1]):
        edges = flatten(list(map(lambda s: s.edges, stage.states)))

        for i in range(len(received_symbols[0])):
            input_bit = received_symbols[k, i]
            if input_bit == 0:
                zero_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input == input_bit, edges)))
                ones_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input != input_bit, edges)))
            else:
                ones_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input == input_bit, edges)))
                zero_edges = list(map(lambda e: e.lmbda, filter(lambda e: e.edge_input != input_bit, edges)))

            if len(zero_edges) == 1 and len(ones_edges) == 1:
                LLRs[k, i] = max_star(zero_edges[0], -np.infty) - \
                    max_star(zero_edges[0], -np.infty) - symbol_bit_LLRs[k, i]
                continue
            if len(ones_edges) == 0 and len(zero_edges) != 0:
                LLRs[k, i] = max_star_recursive(zero_edges) - symbol_bit_LLRs[k, i]
                continue

            LLRs[k, i] = max_star_recursive(zero_edges) - max_star_recursive(ones_edges) - symbol_bit_LLRs[k, i]

    return LLRs


def predict(trellis, received_sequence, LOG_BCJR=True, Es=10, N0=1, verbose=False):
    time_steps = len(received_sequence)

    alpha = np.zeros((trellis.num_states, time_steps + 1))
    beta = np.zeros((trellis.num_states, time_steps + 1))

    # Calculate alphas, betas, gammas and LLRs
    gammas = calculate_gammas(trellis, received_sequence, trellis.num_output_bits, Es, N0, log_bcjr=LOG_BCJR)
    alpha = calculate_alphas(trellis, alpha, log_bcjr=LOG_BCJR)
    beta = calculate_betas(trellis, beta, log_bcjr=LOG_BCJR)
    LLR = calculate_LLRs(trellis, alpha, beta, log_bcjr=LOG_BCJR)

    if verbose:
        print('Message decoded')
    u_hat = np.array([1 if l >= 0 else 0 for l in LLR])

    return u_hat


def ppm_symbols_to_bit_array(received_symbols: npt.ArrayLike, m: int = 4) -> npt.NDArray[np.uint8]:
    """Map PPM symbols back to bit array. """
    received_symbols = np.array(received_symbols)
    reshaped_ppm_symbols = received_symbols.astype(np.uint8).reshape(received_symbols.shape[0], 1)
    bits_array = np.unpackbits(reshaped_ppm_symbols, axis=1)
    received_sequence = bits_array[:, -m:].reshape(-1)

    return received_sequence
