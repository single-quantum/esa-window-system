# %%
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import TimeTagger
from numpy.random import default_rng
from PIL import Image
from scipy.signal import find_peaks

from BCJR_decoder_functions import ppm_symbols_to_bit_array
from core.demodulation_functions import demodulate
from core.encoder_functions import map_PPM_symbols
from ppm_parameters import (CODE_RATE, GREYSCALE, IMG_SHAPE, M, slot_length, num_samples_per_slot,
                            sample_size_awg, num_slots_per_symbol, symbol_length)
from core.scppm_decoder import DecoderError, decode
from core.trellis import Trellis
from core.utils import flatten


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


def simulate_darkcounts_timestamps(rng, lmbda):
    num_slots = int((time_series[msg_peaks][-1] - time_series[msg_peaks][0]) / slot_length)
    p = rng.poisson(lmbda, num_slots)

    darkcounts_timestamps = []
    t0 = time_series[msg_peaks][0]
    for slot_idx, num_events in enumerate(p):
        if num_events == 0:
            continue
        slot_start = t0 + slot_idx * slot_length
        slot_end = t0 + (slot_idx + 1) * slot_length

        darkcounts = rng.uniform(slot_start, slot_end, num_events)
        darkcounts_timestamps.append(darkcounts)

    darkcounts_timestamps = np.array(flatten(darkcounts_timestamps))

    return darkcounts_timestamps


simulate_noise_peaks: bool = True
simulate_lost_symbols: bool = True
simulate_darkcounts: bool = False
simulate_jitter: bool = True

detection_efficiency: float = 0.8
num_photons_per_pulse = 5
darkcounts_factor: float = 0.05
detector_jitter = 125E-12

use_test_file: bool = True
use_latest_tt_file: bool = False
compare_with_original: bool = False
plot_BER_distribution: bool = False

time_events_filename: str
reference_file_path = f'jupiter_greyscale_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence'

if use_test_file:
    time_events_filename = f'ppm_message_Jupiter_tiny_greyscale_{IMG_SHAPE[0]}x{IMG_SHAPE[1]}_pixels_{M}-PPM_{num_samples_per_slot}_3_c1b1_2-3-code-rate.csv'
elif not use_test_file and use_latest_tt_file:
    tt_files_dir = 'time tagger files/'
    tt_files_path = Path(__file__).parent.absolute() / tt_files_dir
    tt_files = tt_files_path.rglob('*.ttbin')
    files: list[Path] = [x for x in tt_files if x.is_file()]
    files = sorted(files, key=lambda x: x.lstat().st_mtime)
    time_events_filename = tt_files_dir + re.split(r'\.\d{1}', files[-1].stem)[0] + '.ttbin'
else:
    time_events_filename = "time tagger files/jupiter_tiny_greyscale_16_samples_per_slot_CSM_0_interleaved_15-56-53.ttbin"

slot_width_ns = num_samples_per_slot * sample_size_awg / 1000
symbol_length_ns = num_slots_per_symbol * slot_width_ns

print_header('PPM parameters')
print_parameter('M (PPM order)', M)
print_parameter('Slot width (ns)', round(slot_width_ns, 3))
print_parameter('Symbol length (ns)', round(symbol_length_ns, 3))
print_parameter('Theoretical countrate (MHz)', 1 / (symbol_length_ns * 1E-9) * 1E-6)

print_parameter('Number of bits per symbol', int(np.log2(M)))
print_parameter('Number of guard slots', M // 4)
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

msg_peaks: npt.NDArray[np.int_] = np.array([])
time_series: npt.NDArray[np.float_] = np.array([])
peak_locations: npt.NDArray[np.float_] = np.array([])

if use_test_file:
    print(f'Decoding file: {time_events_filename}')
    samples = pd.read_csv(time_events_filename, header=None)
    samples = samples.to_numpy().flatten()

    # Make a time series based on the length of samples and how long one sample is in time
    time_series_end = len(samples) * sample_size_awg * 1E-12
    time_series = np.arange(0, time_series_end, sample_size_awg * 1E-12)

    msg_peaks = find_peaks(samples, height=1, distance=2)[0]
else:
    detector_countrate = 80E6
    time_tagger_window_size = 50E-3
    num_events = detector_countrate * time_tagger_window_size
    fr = TimeTagger.FileReader(time_events_filename)
    data = fr.getData(num_events)
    time_stamps = data.getTimestamps()
    peak_locations = time_stamps * 1E-12

    print(f'Number of events: {len(time_stamps)}')

detection_efficiencies = np.arange(0.90, 1.00, 0.05)
cached_trellis: Trellis | None = None

cached_trellis_file_path = Path('cached_trellis_80640_timesteps')
if cached_trellis_file_path.is_file():
    with open('cached_trellis_80640_timesteps', 'rb') as f:
        cached_trellis = pickle.load(f)

for df, detection_efficiency in enumerate(detection_efficiencies):
    irrecoverable: int = 0
    BERS_after = []
    BERS_before = []
    SNRs = []

    for z in range(0, 10):
        print(f'num irrecoverable messages: {irrecoverable}')
        if irrecoverable > 3:
            raise StopIteration("Too many irrecoverable messages. ")
        SEED = 21189 + z**2
        print('Seed', SEED, 'z', z)
        rng = default_rng(SEED)
        np.random.seed(SEED)

        num_darkcounts: int = 0
        if use_test_file:
            # Simulate noise peaks before start and after end of message
            if simulate_noise_peaks:
                noise_peaks = np.sort(rng.integers(0, msg_peaks[0], 15))
                noise_peaks[0] += 1
                noise_peaks_end = np.sort(rng.integers(msg_peaks[-1], len(time_series), 15))
                peaks = np.hstack((noise_peaks, msg_peaks, noise_peaks_end))
            else:
                peaks = msg_peaks

            if simulate_lost_symbols:
                peaks = simulate_symbol_loss(peaks, num_photons_per_pulse, detection_efficiency, seed=SEED)

            num_symbols_received = len(peaks)

            timestamps = time_series[peaks]
            if simulate_darkcounts:
                darkcounts_timestamps = simulate_darkcounts_timestamps(rng, 0.01)
                timestamps = np.sort(np.hstack((timestamps, darkcounts_timestamps)))

            # timestamps = np.hstack((timestamps, rng.random(size=15) * timestamps[0]))
            # timestamps = np.sort(timestamps)

            if simulate_jitter:
                sigma = detector_jitter / 2.355
                timestamps += rng.normal(0, sigma, size=len(timestamps))

            if simulate_darkcounts and num_darkcounts > 0:
                SNR = 10 * np.log10(num_symbols_received / num_darkcounts)
                SNRs.append(SNR)
                print('Signal: ', num_symbols_received, 'Noise: ', num_darkcounts, 'SNR: ', SNR)

            peak_locations = timestamps
            # peak_locations = np.hstack((peak_locations, peak_locations[:len(peak_locations) // 2] + 0.1 * slot_length))
            peak_locations = np.sort(peak_locations)

        try:
            slot_mapped_message = demodulate(peak_locations[:200000], M, slot_length, symbol_length, num_slots_per_symbol, debug_mode=True)
        except ValueError as e:
            irrecoverable += 1
            print(e)
            print('Zero not found in CSM indexes')
            continue

        information_blocks: npt.NDArray[np.int_] = np.array([])
        BER_before_decoding: float | None = None

        try:
            information_blocks, BER_before_decoding = decode(
                slot_mapped_message, M, CODE_RATE,
                **{
                    'use_cached_trellis': True,
                    'cached_trellis_file_path': cached_trellis_file_path,
                    'cached_trellis': cached_trellis,
                    'user_settings': {'reference_file_path': reference_file_path}
                })
        except DecoderError as e:
            print(e)
            print(f'Something went wrong. Seed: {SEED} (z={z})')
            irrecoverable += 1

        BERS_before.append(BER_before_decoding)

        if GREYSCALE:
            pixel_values = map_PPM_symbols(information_blocks, 8)
            try:
                img_arr = pixel_values[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
                CMAP = 'Greys'
                MODE = "L"
                IMG_MODE = 'L'
            except ValueError as e:
                print(e)
                continue
        else:
            img_arr = information_blocks.flatten()[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
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

        if len(information_blocks) < len(sent_message):
            BER_after_decoding = np.sum(np.abs(information_blocks -
                                        sent_message[:len(information_blocks)])) / len(information_blocks)
        else:
            BER_after_decoding = np.sum(
                np.abs(information_blocks[:len(sent_message)] - sent_message)) / len(sent_message)

        if use_test_file and simulate_darkcounts:
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
        # filename = f'BER simulation/decoded_img_64_samples_per_bin_interleaved_' +\
        # f'{num_symbols_lost}_symbols_lost_seed_{SEED}_random_fill.png'
        # # plt.savefig(filename)
        # plt.show()

        # print()
    BERS_after_arr: npt.NDArray[np.float_] = np.array(BERS_after, dtype=float)
    BERS_before_arr: npt.NDArray[np.float_] = np.array(BERS_before, dtype=float)

    # BERS_after = BERS_after[np.where(BERS_after <= 3 * np.std(BERS_after))[0]]
    if plot_BER_distribution:
        plt.figure()
        plt.hist(BERS_before_arr, label='Before decoding')
        plt.hist(BERS_after_arr, label='After decoding')
        plt.title('BER before and after decoding (10% darkcounts)')
        plt.ylabel('Occurences')
        plt.xlabel('Bit Error Ratio (-)')
        plt.legend()
        plt.show()

    bit_error_ratios_after.append(np.mean(BERS_after_arr))
    bit_error_ratios_after_std.append(np.std(BERS_after_arr))

    bit_error_ratios_before.append(np.mean(BERS_before_arr))
    bit_error_ratios_before_std.append(np.std(BERS_before_arr))
    mean_SNRs.append(np.mean(SNRs))

print(f'Average BER before decoding: {bit_error_ratios_before} (std: {bit_error_ratios_before_std})')
print(f'Average BER after decoding: {bit_error_ratios_after} (std: {bit_error_ratios_after_std})')

fig, axs = plt.subplots(1)
axs.errorbar(
    detection_efficiencies, bit_error_ratios_before,
    bit_error_ratios_after_std, 0,
    capsize=2,
    label='Before decoding',
    marker='o',
    markersize=5)

axs.errorbar(
    detection_efficiencies, bit_error_ratios_after,
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
    'darkcounts_factor': darkcounts_factor,
    'detector_jitter': detector_jitter,
    'detection_efficiencies': detection_efficiencies,
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

# %%
