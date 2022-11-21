import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from charset_normalizer import CharsetMatch
from numpy.fft import fft
from scipy.signal import convolve, correlate, find_peaks
from tqdm import tqdm

from BCJR_decoder_functions import ppm_symbols_to_bit_array, predict
from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import (CCSDS_ASM, CSM, ccsds_asm_bit_sequence,
                            num_bins_per_symbol)
from trellis import Trellis
from utils import AWGN, bpsk_encoding, generate_outer_code_edges


def check_csm(symbols):
    check_csm = False
    csm_symbols = np.round(symbols).astype(int)
    csm_sequence = ppm_symbols_to_bit_array(np.array(csm_symbols))

    Es = 10
    N0 = 1

    try:
        berts_sum = sum([1 if x == CSM[i] else 0 for (i, x) in enumerate(csm_symbols)])
    except IndexError as e:
        print(e)

    if berts_sum >= 10:
        check_csm = True

    return check_csm


def check_asm(symbols, trellis):
    check_asm = False
    asm_symbols = np.round(symbols).astype(int)
    # asm_symbols = np.append(asm_symbols, 0)
    asm_sequence = ppm_symbols_to_bit_array(np.array(asm_symbols))

    Es = 5
    N0 = 1

    sigma = np.sqrt(1 / (2 * 3 * Es / N0))

    encoded_sequence = bpsk_encoding(asm_sequence.astype(float))
    encoded_sequence = AWGN(encoded_sequence, sigma)

    u_hat = predict(trellis, encoded_sequence, Es=Es)

    conv = convolve(ccsds_asm_bit_sequence + 0.1, u_hat + 0.1)
    diff = np.sum(np.abs(ccsds_asm_bit_sequence - u_hat[:len(ccsds_asm_bit_sequence)]))

    berts_sum = sum([1 if x == ccsds_asm_bit_sequence[i] else 0 for (i, x)
                    in enumerate(u_hat[:len(ccsds_asm_bit_sequence)])])

    if berts_sum >= 29:

        print(np.max(conv))
        check_asm = True

    return check_asm


def find_msg_indexes(time_stamps, ASM, symbol_length) -> npt.NDArray:
    """Find out where the message starts, given that there is some space between messages, where only noise is received.

    Timestamps should be in seconds. """
    noise_peaks = np.where(np.diff(time_stamps) / symbol_length > 40)[0]
    msg_start_idxs = noise_peaks[np.where(np.diff(noise_peaks) > 3780)[0]]

    return msg_start_idxs


def find_msg_start(estimated_msg_start_idxs, time_stamps, ASM, bin_length, symbol_length):
    j = 0
    msg_start_idxs = []

    memory_size = 2
    num_output_bits = 3
    num_input_bits = 1
    edges = generate_outer_code_edges(memory_size, bpsk_encoding=True)
    Es = 5
    N0 = 1

    time_steps = int(len(CSM) * 4 / 3)
    tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
    tr.set_edges(edges)

    while j < len(estimated_msg_start_idxs) - 1:
        asm_found = False
        shift = 0
        while not asm_found:
            n0 = estimated_msg_start_idxs[j] + shift
            ne = estimated_msg_start_idxs[j + 1] + shift

            if ne - n0 < len(CSM):
                break

            # The first peak occurs at t=0. All other pulse positions are measured relative to this peak.
            t0 = time_stamps[n0]

            shifted_time_stamps = np.array(time_stamps[n0:ne] - t0) + CSM[0] * bin_length

            symbols, _ = parse_ppm_symbols(shifted_time_stamps[:len(CSM) + 1], bin_length, symbol_length)

            asm_found: bool = check_csm(symbols[:len(CSM)])

            shift += 1
            if shift > 10 and not asm_found:
                break

            if asm_found:
                msg_start_idxs.append(n0)
        j += 1

    return msg_start_idxs


def find_asm_idxs(time_stamps, ASM, bin_length, symbol_length):
    asm_idxs = []

    memory_size = 2
    num_output_bits = 3
    num_input_bits = 1
    edges = generate_outer_code_edges(memory_size, bpsk_encoding=True)
    Es = 5
    N0 = 1

    time_steps = int(len(ASM) * 4 / 3)
    tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
    tr.set_edges(edges)

    i = 0
    while i < len(time_stamps) - len(ASM) and len(asm_idxs) < 3:
        if i % 500 == 0:
            print('i', i)

        symbols, _, _ = parse_ppm_symbols(time_stamps[i:i + len(ASM) + 1] - time_stamps[i], bin_length, symbol_length)

        asm_found: bool = check_asm(symbols, tr)

        if asm_found:
            print('Found ASM at idx', i)
            asm_idxs.append(i)

            i += 15000

        i += 1

    return asm_idxs


def find_csm_idxs(time_stamps, CSM, bin_length, symbol_length):
    csm_idxs = []
    csms_found_with_correlation = []

    i = 0
    darkcounts = 0
    CSM_bin_distances = np.diff([CSM[i] + i * num_bins_per_symbol for i in range(len(CSM))])
    sr = 2000
    N = len(CSM_bin_distances)
    n = np.arange(N)
    T = N / sr
    freq = n / T
    X = fft(CSM_bin_distances)

    # symbol_bin_distances = np.round(np.diff(time_stamps) / bin_length)
    # corr = correlate(symbol_bin_distances, CSM_bin_distances)
    # possible_csm_idxs = find_peaks(corr, height=6800, distance=3700)[0]
    plt.figure()
    plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.show()
    print('FFT X', np.abs(X)[np.array([2, 5, 6, 9, 10, 13])])

    while i < len(time_stamps) - len(CSM):
        symbol_bin_distances = np.round(np.diff(time_stamps[i:i + len(CSM)]) / bin_length)
        corr = correlate(CSM_bin_distances, symbol_bin_distances)
        corr_coeff = np.corrcoef(CSM_bin_distances, symbol_bin_distances)[0, 1]

        if i == 52128:
            xyz = 1
            Y = fft(symbol_bin_distances)
            plt.figure()
            plt.stem(freq, np.abs(Y), 'b', markerfmt=" ", basefmt="-b")
            plt.xlabel('Freq (Hz)')
            plt.ylabel('FFT Amplitude |X(freq)|')
            plt.show()
            print('FFT Y', np.abs(Y)[np.array([2, 5, 6, 9, 10, 13])])
        if corr_coeff > 0.92:
            csms_found_with_correlation.append(i)

        symbols, _, read_idxs = parse_ppm_symbols(
            time_stamps[i:i + len(CSM)] - time_stamps[i], bin_length, symbol_length)
        if len(symbols) > len(CSM):
            symbols = symbols[:len(CSM)]
        darkcounts += (read_idxs[0] - read_idxs[1])
        csm_found: bool = check_csm(symbols)

        if csm_found:
            print('Found CSM at idx', i)
            csm_idxs.append(i)
            # symbols, _, read_idxs = parse_ppm_symbols(time_stamps[i:i+2*3780]-time_stamps[i], bin_length, symbol_length)
            # x, y, z = parse_ppm_symbols(time_stamps[i+3779:i+2*3780]-time_stamps[i+3779], bin_length, symbol_length)
            i += int(0.7 * 15120 // 4)

        i += 1

    return csm_idxs
