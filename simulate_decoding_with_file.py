# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import TimeTagger
from numpy.random import default_rng
from PIL import Image
from scipy.signal import find_peaks

from BCJR_decoder_functions import ppm_symbols_to_bit_array, predict
from demodulation_functions import find_csm_idxs, find_msg_indexes
from encoder_functions import (bit_deinterleave, channel_deinterleave,
                               map_PPM_symbols)
from parse_ppm_symbols import parse_ppm_symbols
from ppm_parameters import (BIT_INTERLEAVE, CHANNEL_INTERLEAVE, CSM, GREYSCALE,
                            IMG_SHAPE, B_interleaver, M, N_interleaver,
                            bin_length, m, num_bins_per_symbol,
                            num_samples_per_slot, sample_size_awg,
                            symbol_length, symbols_per_codeword)
from trellis import Trellis
from utils import bpsk_encoding, flatten, generate_outer_code_edges
from time import time
from datetime import datetime

def print_parameter(parameter_str: str, parameter, spacing: int = 30):
    print(f'{parameter_str:<{spacing}} {parameter}')


def print_header(header: str, len_header: int = 50, filler='-'):
    len_filler = (len_header - len(header)) // 2 - 2
    if len(header) % 2 == 0:
        print(f'{"#":{filler}<{len_filler}} {header} {"#" :{filler}>{len_filler}}')
    else:
        print(f'{"#":{filler}<{len_filler}} {header} {"#" :{filler}>{len_filler+1}}')


def simulate_symbol_loss(
        peaks: npt.NDArray,
        num_photons_per_pulse: int,
        detection_efficiency: float,
        seed: int = 777) -> npt.NDArray:
    """ Simulate the loss of symbols, based on the number of photons per pulse and detection efficiency.

    For each symbol, use the poisson distribution to determine how many photons arrived in each symbol pulse
    Then, do n bernoulli trials (binomial distribution) and success probability p, where n the number of photons
    per pulse and p is the detection efficiency. """

    rng = default_rng(seed)
    num_symbols = len(peaks)

    num_photons_detected_per_pulse = rng.binomial(
        rng.poisson(num_photons_per_pulse, size=num_symbols),
        detection_efficiency)

    idxs_to_be_removed = np.where(num_photons_detected_per_pulse == 0)[0]
    peaks = np.delete(peaks, idxs_to_be_removed)

    return peaks


def new_method(csm_idxs, peak_locations):
    len_codeword = symbols_per_codeword + len(CSM)
    len_codeword_no_CSM = symbols_per_codeword

    msg_symbols = []

    # Method 1
    for i in range(len(csm_idxs) - 1):
        start = n0 + csm_idxs[i]
        stop = n0 + csm_idxs[i + 1]
        t0_codeword = peak_locations[start]
        fraction_lost = (peak_locations[stop] - peak_locations[start]) / (symbol_length * len_codeword) - 1
        num_codewords_lost = round(fraction_lost)

        start = n0 + csm_idxs[i]
        stop = n0 + csm_idxs[i + 1]

        symbols, _ = parse_ppm_symbols(peak_locations[start:stop] - t0_codeword, bin_length, symbol_length)

        if len(symbols) < len_codeword:
            diff = len_codeword - len(symbols)
            symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        if num_codewords_lost >= 1:
            csm_estimates_to_delete = flatten(
                [range(i * len_codeword, i * len_codeword + len(CSM)) for i in range(1, num_codewords_lost + 1)])
            symbols = np.delete(symbols, csm_estimates_to_delete)

            diff = (num_codewords_lost + 1) * len_codeword_no_CSM - (len(symbols) - len(CSM))
            if diff > 1:
                symbols = np.hstack((symbols, np.random.randint(0, M, diff)))

        msg_symbols.append(np.round(symbols[len(CSM):]).astype(int))

    t0_codeword = peak_locations[n0 + csm_idxs[-1]]
    symbols, _ = parse_ppm_symbols(peak_locations[n0 + csm_idxs[-1]:ne] - t0_codeword, bin_length, symbol_length)
    msg_symbols.append(np.round(symbols[len(CSM):]).astype(int))

    return msg_symbols


simulate_noise_peaks: bool = True
simulate_lost_symbols: bool = False
simulate_darkcounts: bool = True
simulate_jitter: bool = False

detection_efficiency: float = 0.8
num_photons_per_pulse = 10
darkcounts_factor: float = 0.05
detector_jitter = 50E-12

use_test_file: bool = True
compare_with_original: bool = False
plot_BER_distribution: bool = False

slot_width_ns = num_samples_per_slot * sample_size_awg / 1000
symbol_length_ns = num_bins_per_symbol * slot_width_ns

print_header('PPM parameters')
print_parameter('M (PPM order)', M)
print_parameter('Slot width (ns)', round(slot_width_ns, 3))
print_parameter('Symbol length (ns)', round(symbol_length_ns, 3))
print_parameter('Theoretical countrate (MHz)', 1 / (symbol_length_ns * 1E-9) * 1E-6)

print_parameter('Number of guard slots', int(np.log2(M)))
print_header("-")

print()

print_header('Detector parameters')
print_parameter('Detection efficiency', detection_efficiency * 100)
print_parameter('Timing jitter (ps)', detector_jitter * 1E12)
print_header("-")

# Seed that causes a CSM in the middle to be lost: 889
# Seed that causes a CSM at the end to be lost: 777
# both with cutting out 4500 symbols
# SEED = 889
bit_error_ratios_before = []
bit_error_ratios_after = []

bit_error_ratios_after_std = []
bit_error_ratios_before_std = []

# The simulation is repeated a couple of times, so take the mean SNR for each set of simulations.
mean_SNRs = []

symbols_lost_lower_bound = 5000
symbols_lost_upper_bound = 8000

# Trellis paramters (can be defined outside of for loop for optimisation)
num_output_bits: int = 3
num_input_bits: int = 1
memory_size: int = 2
edges = generate_outer_code_edges(memory_size, bpsk_encoding=False)

if use_test_file:
    samples = pd.read_csv('ppm_message_Jupiter_tiny_greyscale_95x100_pixels_8-PPM_8_3_c0b0.csv', header=None)
    samples = samples.to_numpy().flatten()

    # Make a time series based on the length of samples and how long one sample is in time
    time_series_end = len(samples) * sample_size_awg * 1E-12
    time_series = np.arange(0, time_series_end, sample_size_awg * 1E-12)

    msg_peaks = find_peaks(samples, height=1, distance=10)[0]
else:
    detector_countrate = 12.7E6
    time_tagger_window_size = 50E-3
    num_events = detector_countrate * time_tagger_window_size
    fr = TimeTagger.FileReader(
        "time tagger files/10-25-2022/jupiter_tiny_greyscale_64_samples_per_slot_CSM_65-0_interleaved_17-41-00.ttbin")
    data = fr.getData(num_events)
    time_stamps = data.getTimestamps()
    peak_locations = time_stamps * 1E-12

    print(f'Number of events: {len(time_stamps)}')

darkcounts_factors = np.logspace(-2, -0.3, num=15)

for darkcounts_factor in darkcounts_factors:
    BERS_after = []
    BERS_before = []
    SNRs = []

    for z in range(0, 15):
        SEED = 21189 + z**2
        print('Seed', SEED, 'z', z)
        rng = default_rng(SEED)
        np.random.seed(SEED)

        if use_test_file:
            # Simulate noise peaks before start and after end of message
            if simulate_noise_peaks:
                noise_peaks = np.sort(rng.integers(0, msg_peaks[0], 15))
                noise_peaks_end = np.sort(rng.integers(msg_peaks[-1], len(time_series), 15))
                peaks = np.hstack((noise_peaks, msg_peaks, noise_peaks_end))
            else:
                peaks = msg_peaks

            if simulate_lost_symbols:
                peaks = simulate_symbol_loss(peaks, num_photons_per_pulse, detection_efficiency, seed=SEED)

            num_symbols_received = len(peaks)

            if simulate_darkcounts:
                num_darkcounts = int(len(msg_peaks) * darkcounts_factor)
                # To make sure we have a unique set of integers, create a range and do a choice from that.
                msg_idx_range = np.arange(np.min(msg_peaks), np.max(msg_peaks), 1)
                darkcount_indexes = np.random.choice(msg_idx_range, num_darkcounts)
                peaks = np.sort(np.hstack((peaks, darkcount_indexes)))
            timestamps = time_series[peaks]

            if simulate_jitter:
                sigma = detector_jitter / 2.355
                timestamps += rng.normal(0, sigma, size=len(timestamps))

            if simulate_darkcounts and num_darkcounts > 0:
                SNR = 10 * np.log10(num_symbols_received / num_darkcounts)
                SNRs.append(SNR)
                print('Signal: ', num_symbols_received, 'Noise: ', num_darkcounts, 'SNR: ', SNR)

            n0 = 0
            t0 = timestamps[n0]
            shifted_time_stamps = np.array(timestamps - t0) + CSM[0] * bin_length
            peak_locations = timestamps + 0.1 * bin_length

        estimated_msg_start_idxs = find_msg_indexes(peak_locations, CSM, symbol_length)
        n0 = estimated_msg_start_idxs[0]
        if use_test_file:
            ne = len(peaks)
        else:
            ne = estimated_msg_start_idxs[1]
        # ne = len(peak_locations)
        j = 0
        symbol_distance = np.diff(peak_locations[n0 + j:ne + j]) / symbol_length
        while np.mean(symbol_distance[0:4]) > 3:
            j += 1
            symbol_distance = np.diff(peak_locations[n0 + j:ne + j]) / symbol_length
            if j > 15:
                raise StopIteration("Could not find msg start")

        je = 1
        while np.mean(symbol_distance[-5 - je:-je]) > 1.5:
            je += 1
            if je > 20000:
                raise StopIteration("Could not find msg end")

        n0 += j
        ne -= je

        print(f'Number of detection events in message frame: {len(peak_locations[n0:ne])}')

        t0_msg = peak_locations[n0] + CSM[0] * bin_length
        t_end = peak_locations[-n0 - 1]

        csm_idxs = find_csm_idxs(peak_locations[n0:ne] - t0_msg, CSM, bin_length, symbol_length)
        if csm_idxs[0] > 5:
            csm_idxs.append(0)
            csm_idxs = np.sort(csm_idxs)
            print('Zero not found in CSM indexes')


        msg_symbols = new_method(csm_idxs, peak_locations)

        new = np.array(flatten(msg_symbols))

        ppm_mapped_message = new

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
            BER_before_decoding = np.sum(np.abs(convoluted_bit_sequence[:len(
                sent_bit_sequence)] - sent_bit_sequence)) / len(sent_bit_sequence)
        else:
            BER_before_decoding = np.sum(np.abs(convoluted_bit_sequence -
                                         sent_bit_sequence[:len(convoluted_bit_sequence)])) / len(sent_bit_sequence)

        print(f'BER before decoding: {BER_before_decoding}')
        if BER_before_decoding > 0.25:
            print(f'Something went wrong. Seed: {SEED} (z={z})')
            continue

        BERS_before.append(BER_before_decoding)
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

        num_states = 2**memory_size
        time_steps = deinterleaved_received_sequence_2.shape[0] // num_output_bits
        
        start = time()
        tr = Trellis(memory_size, num_output_bits, time_steps, edges, num_input_bits)
        tr.set_edges(edges)
        end = time()
        print('Set edges run time', end-start)

        Es = 5
        N0 = 1
        sigma = np.sqrt(1 / (2 * 3 * Es / N0))

        encoded_sequence_2 = bpsk_encoding(deinterleaved_received_sequence_2.astype(float))
        # encoded_sequence_2 = AWGN(encoded_sequence_2, sigma)

        alpha = np.zeros((num_states, time_steps + 1))
        beta = np.zeros((num_states, time_steps + 1))

        predicted_msg = predict(tr, encoded_sequence_2, Es=Es)
        termination_bits_removed = predicted_msg.reshape((-1, 5040))[:, :-2].flatten()

        if GREYSCALE:
            pixel_values = map_PPM_symbols(termination_bits_removed, 8)
            # img_arr = pixel_values[:IMG_SHAPE[0]*IMG_SHAPE[1]].reshape(IMG_SHAPE)
            img_arr = pixel_values[:9400].reshape((94, 100))
            CMAP = 'Greys'
            MODE = "L"
            IMG_MODE = 'L'
        else:
            img_arr = predicted_msg[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
            CMAP = 'binary'
            MODE = "1"
            IMG_MODE = '1'

        # compare to original image
        file = "sample_payloads/JWST_2022-07-27_Jupiter_tiny.png"
        img = Image.open(file)
        img = img.convert(IMG_MODE)
        sent_img_array = np.asarray(img).astype(int)

        img_shape = sent_img_array.shape
        # In the case of greyscale, each pixel has a value from 0 to 255.
        # This would be the same as saying that each pixel is a symbol, which should be mapped to an 8 bit sequence.
        if GREYSCALE:
            sent_message = ppm_symbols_to_bit_array(sent_img_array.flatten(), 8)
        else:
            sent_message = sent_img_array.flatten()

        if len(termination_bits_removed) < len(sent_message):
            BER_after_decoding = np.sum(np.abs(termination_bits_removed -
                                        sent_message[:len(termination_bits_removed)])) / len(termination_bits_removed)
        else:
            BER_after_decoding = np.sum(
                np.abs(termination_bits_removed[:len(sent_message)] - sent_message)) / len(sent_message)
        
        if simulate_darkcounts:
            print(f'BER after decoding: {BER_after_decoding }. Number of darkcounts: {num_darkcounts}')
        else:
            print(f'BER after decoding: {BER_after_decoding }. ')


        BERS_after.append(BER_after_decoding)

        if compare_with_original:
            fig, axs = plt.subplots(1, 2, figsize=(5, 4))
            plt.suptitle('Detector B')
            axs[0].imshow(sent_img_array, cmap=CMAP)
            axs[0].set_xlabel('Pixel number (x)')
            axs[0].set_ylabel('Pixel number (y)')
            axs[0].set_title('Original image')

            axs[1].imshow(img_arr, cmap=CMAP)
            axs[1].set_xlabel('Pixel number (x)')
            axs[1].set_ylabel('Pixel number (y)')
            axs[1].set_title('Decoded image')
            plt.show()

        # plt.figure()
        # plt.imshow(img_arr)
        # plt.title('Decoded image of Jupiter (with bit and channel interleaving)')
        # plt.xlabel('Pixel number (x)')
        # plt.ylabel('Pixel number (y)')
        # plt.text(x=2, y=90, s=f'BER={BER_after_decoding:.3f}', ha='left',
        #          va='center', color='magenta', size=13, weight='bold')
        # # plt.savefig(f'BER simulation/decoded_img_64_samples_per_bin_interleaved_{num_symbols_lost}_symbols_lost_seed_{SEED}_random_fill.png')
        # plt.show()

        # print()
    BERS_after = np.array(BERS_after)
    BERS_before = np.array(BERS_before)

    # BERS_after = BERS_after[np.where(BERS_after <= 3 * np.std(BERS_after))[0]]
    if plot_BER_distribution:
        plt.figure()
        plt.hist(BERS_before, label='Before decoding')
        plt.hist(BERS_after, label='After decoding')
        plt.title('BER before and after decoding (10% darkcounts)')
        plt.ylabel('Occurences')
        plt.xlabel('Bit Error Ratio (-)')
        plt.legend()
        plt.show()

    bit_error_ratios_after.append(np.mean(BERS_after))
    bit_error_ratios_after_std.append(np.std(BERS_after))

    bit_error_ratios_before.append(np.mean(BERS_before))
    bit_error_ratios_before_std.append(np.std(BERS_before))
    mean_SNRs.append(np.mean(SNRs))

fig, axs = plt.subplots(1)
axs.errorbar(
    mean_SNRs, bit_error_ratios_before,
    bit_error_ratios_after_std, 0, 
    capsize=2, 
    label='Before decoding', 
    marker='o',
    markersize=5)

axs.errorbar(
    mean_SNRs, bit_error_ratios_after, 
    bit_error_ratios_before_std, 0, 
    capsize=2, 
    label='After decoding', 
    marker='o',
    markersize=5)

axs.set_yscale('log')
axs.set_ylabel('Bit Error Ratio (-)')
axs.set_xlabel('Signal to Noise Ratio (-)')
plt.title('BER as function of SNR')
plt.legend()
plt.show()

print('done')
print()

log = {
    'simulate_noise_peaks': simulate_noise_peaks,
    'simulate_lost_symbols': simulate_lost_symbols,
    'simulate_darkcounts': simulate_darkcounts,
    'simulate_jitter': simulate_jitter,
    'detection_efficiency': detection_efficiency,
    'num_photons_per_pulse': num_photons_per_pulse,
    'darkcounts_factor':  darkcounts_factor,
    'detector_jitter': detector_jitter,
    'darkcounts_factors': darkcounts_factors,
    'data': {
        'bit_error_ratios_after': bit_error_ratios_after,
        'bit_error_ratios_before': bit_error_ratios_before,
        'bit_error_ratios_after_std': bit_error_ratios_after_std,
        'bit_error_ratios_before_std': bit_error_ratios_before_std
    }
}

now = datetime.now().strftime('%d-%m-%Y')
with open(f'var_dump_simulate_decoding_with_file_{now}', 'wb') as f:
    pickle.dump(log, f)