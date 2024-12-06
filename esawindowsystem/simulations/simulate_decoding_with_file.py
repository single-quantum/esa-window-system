# %%
import pickle
import re
from datetime import datetime
from pathlib import Path
from fractions import Fraction

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import TimeTagger
from numpy.random import default_rng
from PIL import Image
from scipy.signal import find_peaks

from esawindowsystem.core.BCJR_decoder_functions import \
    ppm_symbols_to_bit_array
from esawindowsystem.core.demodulation_functions import demodulate
from esawindowsystem.core.encoder_functions import map_PPM_symbols, get_asm_bit_arr
from esawindowsystem.core.scppm_decoder import DecoderError, decode
from esawindowsystem.core.trellis import Trellis
from esawindowsystem.core.utils import flatten
from esawindowsystem.generate_awg_pattern import generate_awg_pattern
from esawindowsystem.ppm_parameters import (CODE_RATE, GREYSCALE,
                                            IMG_FILE_PATH, IMG_SHAPE, M,
                                            num_samples_per_slot,
                                            num_slots_per_symbol,
                                            sample_size_awg, slot_length,
                                            symbol_length)
from esawindowsystem.simulations.simulation_utils import (
    get_simulated_message_peak_locations, print_header, print_parameter)


def run_simulation_loop(
    detection_efficiency: float,
    darkcounts_factor: float,
    detector_jitter: float,
    csm_correlation_threshold: float = 0.55,
    num_pixels: int = 1,
    num_photons_per_pulse: int = 1,
    num_repeats: int = 5,
    z0: int = 0,
    detection_events_timestamps=None
):
    if use_test_file:
        # Print the settings used for the simulation
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
        print_parameter('Number of pixels', num_pixels)
        print_header("-")

        print()

        print_header('Simulation settings')
        print_parameter('Number of photons per pulse (-)', num_photons_per_pulse)
        print_parameter('Simulate detector efficiency', simulate_lost_symbols)
        print_parameter('Simulate detector jitter', simulate_jitter)
        print_parameter('Simulate darkcounts', simulate_darkcounts)
        print_header('-')

    irrecoverable: int = 0
    BERS_after = []
    BERS_before = []
    SNRs: list[float] = []

    bit_error_ratios_before: list[float] = []
    bit_error_ratios_after: list[float] = []

    bit_error_ratios_after_std: list[float] = []
    bit_error_ratios_before_std: list[float] = []

    for z in range(z0, num_repeats):
        print()
        print(f'num irrecoverable messages: {irrecoverable}')
        if irrecoverable > 3:
            return ([None], [None], [None], [None])

        SEED = 21190 + z**2
        print('Seed', SEED, 'z', z)
        rng = default_rng(SEED)

        num_darkcounts: int = 0

        # Simulate detection efficiency / jitter / darkcounts
        if use_test_file:
            detection_events_timestamps = get_simulated_message_peak_locations(
                detection_events_indexes, time_series, slot_length, simulate_noise_peaks, simulate_lost_symbols,
                simulate_darkcounts, darkcounts_factor, simulate_jitter, num_photons_per_pulse,
                detection_efficiency, num_pixels, detector_jitter, rng, SEED
            )

            num_symbols_received = len(detection_events_timestamps)

            if simulate_darkcounts and num_darkcounts > 0:
                SNR = 10 * np.log10(num_symbols_received / num_darkcounts)
                SNRs.append(SNR)
                print('Signal: ', num_symbols_received, 'Noise: ', num_darkcounts, 'SNR: ', SNR)

        try:
            slot_mapped_message, num_events_per_slot, _ = demodulate(
                detection_events_timestamps, M, slot_length, symbol_length, csm_correlation_threshold=csm_correlation_threshold, **{'debug_mode': demodulator_debug_mode})
        except ValueError as e:
            irrecoverable += 1
            print(e)
            print('Zero not found in CSM indexes')
            continue

        information_blocks: npt.NDArray[np.int_] = np.array([])
        BER_before_decoding: float | None = None

        with open('sent_bit_sequence', 'rb') as f:
            sent_bits = pickle.load(f)

        with open('sent_bit_sequence_no_csm', 'rb') as f:
            sent_bits_no_csm = pickle.load(f)

        try:
            information_blocks, BER_before_decoding, where_asms = decode(
                slot_mapped_message, M, CODE_RATE, CHANNEL_INTERLEAVE=True, BIT_INTERLEAVE=True, use_inner_encoder=True,
                **{
                    'use_cached_trellis': False,
                    # 'cached_trellis_file_path': cached_trellis_file_path,
                    'user_settings': {'reference_file_path': reference_file_path},
                    'num_events_per_slot': num_events_per_slot,
                    'debug_mode': decoder_debug_mode,
                    'sent_bit_sequence_no_csm': sent_bits_no_csm
                })

            information_block_sizes = {
                Fraction(1, 3): 5040,
                Fraction(1, 2): 7560,
                Fraction(2, 3): 10080
            }

            num_bits = information_block_sizes[CODE_RATE]
            ASM_arr = get_asm_bit_arr()

            information_blocks = information_blocks[where_asms[0] +
                                                    ASM_arr.shape[0]:(where_asms[0] + ASM_arr.shape[0] + num_bits * 8)]

        except DecoderError as e:
            print(e)
            print(f'Something went wrong. Seed: {SEED} (z={z})')
            irrecoverable += 1
            continue

        BERS_before.append(BER_before_decoding)

        # compare to original image
        file = Path.cwd() / IMG_FILE_PATH
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

        print()
        if use_test_file and simulate_darkcounts:
            print(f'BER after decoding: {BER_after_decoding:.3e}. Number of darkcounts: {num_darkcounts}')
        else:
            print(f'BER after decoding: {BER_after_decoding:.3e}. ')

        BERS_after.append(BER_after_decoding)

        if GREYSCALE:
            try:
                pixel_values = map_PPM_symbols(information_blocks, 8)
                img_arr = pixel_values[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
                CMAP = 'Greys'
                MODE = "L"
                IMG_MODE = 'L'
            except ValueError as e:
                print(e)
                print(f'Malformed information blocks. Seed: {SEED} (z={z})')
                irrecoverable += 1
                continue

        else:
            img_arr = information_blocks.flatten()[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
            CMAP = 'binary'
            MODE = "1"
            IMG_MODE = '1'

        difference_mask = np.where(sent_img_array != img_arr)
        highlighted_image = img_arr.copy()
        highlighted_image[difference_mask] = -1

        custom_cmap = mpl.colormaps['Greys']
        custom_cmap.set_under(color='r')

        if compare_with_original:
            label_font_size = 14
            fig, axs = plt.subplots(1, 2, figsize=(5, 4))
            plt.suptitle('SCPPM message comparison', fontsize=18)
            axs[0].imshow(sent_img_array, cmap=CMAP)
            axs[0].set_xlabel('Pixel number (x)', fontsize=label_font_size)
            axs[0].set_ylabel('Pixel number (y)', fontsize=label_font_size)
            axs[0].tick_params(axis='both', which='major', labelsize=label_font_size)
            axs[0].set_title('Original image', fontsize=16)

            axs[1].imshow(highlighted_image, cmap=custom_cmap, vmin=0)
            axs[1].set_xlabel('Pixel number (x)', fontsize=label_font_size)
            axs[1].set_ylabel('Pixel number (y)', fontsize=label_font_size)
            axs[1].tick_params(axis='both', which='major', labelsize=label_font_size)

            axs[1].set_title('Decoded image', fontsize=16)
            plt.show()

        if plot_decoded_image:
            plt.figure()
            plt.imshow(img_arr)
            plt.title('Decoded image of Jupiter (with bit and channel interleaving)')
            plt.xlabel('Pixel number (x)')
            plt.ylabel('Pixel number (y)')
            plt.text(x=2, y=90, s=f'BER={BER_after_decoding:.3f}', ha='left',
                     va='center', color='magenta', size=13, weight='bold')
            filename = 'BER simulation/decoded_img_64_samples_per_bin_interleaved_' +\
                f'{num_symbols_lost}_symbols_lost_seed_{SEED}_random_fill.png'
            if save_decoded_image:
                plt.savefig(filename)
            plt.show()

        # print()
    BERS_after_arr: npt.NDArray[np.float64] = np.array(BERS_after, dtype=float)
    BERS_before_arr: npt.NDArray[np.float64] = np.array(BERS_before, dtype=float)

    # BERS_after = BERS_after[np.where(BERS_after <= 3 * np.std(BERS_after))[0]]
    if plot_BER_distribution:
        plt.figure()
        plt.hist(BERS_before_arr, label='Before decoding')
        plt.hist(BERS_after_arr, label='After decoding')
        plt.title('BER before and after decoding')
        plt.ylabel('Occurences')
        plt.xlabel('Bit Error Ratio (-)')
        plt.legend()
        plt.show()

    # Remove statistical outliers
    for _ in range(2):
        z_score = (np.array(BERS_after_arr) - np.mean(BERS_after_arr)) / np.std(BERS_after_arr)
        outliers = np.where(abs(z_score) > 2)[0]
        BERS_after_arr = np.delete(BERS_after_arr, outliers)

    for _ in range(2):
        z_score = (np.array(BERS_before_arr) - np.mean(BERS_before_arr)) / np.std(BERS_before_arr)
        outliers = np.where(abs(z_score) > 2)[0]
        BERS_before_arr = np.delete(BERS_before_arr, outliers)

    bit_error_ratios_after.append(np.mean(BERS_after_arr))
    bit_error_ratios_after_std.append(np.std(BERS_after_arr))

    bit_error_ratios_before.append(np.mean(BERS_before_arr))
    bit_error_ratios_before_std.append(np.std(BERS_before_arr))
    mean_SNRs.append(np.mean(SNRs))

    return bit_error_ratios_before, bit_error_ratios_before_std, bit_error_ratios_after, bit_error_ratios_after_std


simulate_noise_peaks: bool = True
simulate_lost_symbols: bool = True
simulate_darkcounts: bool = True
simulate_jitter: bool = True
use_test_file: bool = True
use_latest_tt_file: bool = False

z0 = 0      # Used for debugging purposes, to start from a specific iteration.
demodulator_debug_mode: bool = False
decoder_debug_mode: bool = True
csm_correlation_threshold = 0.50

# Detector settings
detection_efficiency_lower: float = 0.45
detection_efficiency_upper: float = 0.45
detection_efficiency_step_size = 0.05
num_photons_per_pulse: list[float] = [4.2]
darkcounts_factor: float = 0.015
detector_jitter = 100E-12
num_pixels = [4, 3]

num_samples_ppm_pulse = 10
num_copies_message: int = 1          # When using CSV test file, use n copies of the same message
num_repeats_simulation = 3          # Repeat simulation N times for statistical purposes

# Plot settings
compare_with_original: bool = True
plot_BER_distribution: bool = False
plot_decoded_image: bool = False
save_decoded_image: bool = False


# Load timestamps from time tagger file or simulate time tags from CSV file
time_events_filename: Path
base_dir = Path.cwd() / Path('esawindowsystem') / Path('ppm_sample_messages')
reference_file_path = Path.cwd(
) / f'herbig_haro_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence'

if use_test_file:
    cr = str(CODE_RATE).replace('/', '-')

    if GREYSCALE:
        time_events_filename = base_dir / Path(f'ppm_message_JWST_Jupiter_greyscale_{IMG_SHAPE[0]}x{IMG_SHAPE[1]}_pixels_{M}' +
                                               f'-PPM_{num_samples_per_slot}_{num_samples_ppm_pulse}_c1b1_{cr}-code-rate_square.csv')
    else:
        time_events_filename = base_dir / Path(f'ppm_message_JWST_Jupiter_{IMG_SHAPE[0]}x{IMG_SHAPE[1]}_pixels_{M}' +
                                               f'-PPM_{num_samples_per_slot}_{num_samples_ppm_pulse}_c1b1_{cr}-code-rate_square.csv')

elif not use_test_file and use_latest_tt_file:
    tt_files_dir = 'time tagger files/'
    # Go up three times. Path(__file__).parent -> esawindowsystem/simulations
    tt_files_path = Path(__file__).parent.parent.parent.absolute() / tt_files_dir
    tt_files = tt_files_path.rglob('*.ttbin')
    files: list[Path] = [x for x in tt_files if x.is_file()]
    files = sorted(files, key=lambda x: x.lstat().st_mtime)
    time_events_filename = tt_files_dir + re.split(r'\.\d{1}', files[-1].stem)[0] + '.ttbin'
else:
    time_events_filename = 'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\C824-08\\16 ppm\\500 ps slots\\JWST_Jupiter_tiny_5-sps_16-PPM_2-3-code-rate_14-35-48_1730900148.ttbin'

slot_width_ns = num_samples_per_slot * sample_size_awg / 1000
symbol_length_ns = num_slots_per_symbol * slot_width_ns

# The simulation is repeated a couple of times, so take the mean SNR for each set of simulations.
mean_SNRs = []

detection_events_indexes: npt.NDArray[np.int_] = np.array([])
time_series: npt.NDArray[np.float64] = np.array([])
peak_locations: npt.NDArray[np.float64] = np.array([])

print(f'Decoding file: {time_events_filename}')

if use_test_file:
    if not Path.is_file(Path(time_events_filename)):
        generate_awg_pattern(num_samples_ppm_pulse, pulse_shape='square')
        time_events_filename = Path('esawindowsystem') / Path('ppm_sample_messages') / Path(time_events_filename)
    samples = pd.read_csv(time_events_filename, header=None)
    samples_np_array: npt.NDArray[np.int_] = samples.to_numpy().flatten()

    samples_np_array = np.tile(samples_np_array, num_copies_message)

    samples_np_array = np.hstack((np.zeros(500, dtype=int), samples_np_array))

    # Make a time series based on the length of samples and how long one sample is in time
    time_series_end = len(samples_np_array) * sample_size_awg * 1E-12
    time_series = np.arange(0, time_series_end, sample_size_awg * 1E-12)

    detection_events_indexes = find_peaks(samples_np_array, height=1, distance=2)[0]
    detection_events_timestamps = None
else:
    detector_countrate = 4.41E7
    time_tagger_window_size = 50E-3
    num_events = detector_countrate * time_tagger_window_size
    fr = TimeTagger.FileReader(time_events_filename)
    data = fr.getData(num_events)
    time_stamps = data.getTimestamps()
    detection_events_timestamps = time_stamps * 1E-12

    print(f'Number of events: {len(detection_events_timestamps)}')


num_efficiency_steps = round((detection_efficiency_upper - detection_efficiency_lower) /
                             detection_efficiency_step_size + 1)
detection_efficiencies = np.linspace(detection_efficiency_lower, detection_efficiency_upper, num_efficiency_steps)

cached_trellis: Trellis | None = None

cached_trellis_file_path = Path('cached_trellis_80640_timesteps')
# if cached_trellis_file_path.is_file():
#     with open('cached_trellis_80640_timesteps', 'rb') as f:
#         cached_trellis = pickle.load(f)

simulation_data_dict = {}

for nppp in num_photons_per_pulse:
    for pixels in num_pixels:

        bit_error_ratios_before, bit_error_ratios_before_std, bit_error_ratios_after, bit_error_ratios_after_std = run_simulation_loop(
            detection_efficiency_lower, darkcounts_factor, detector_jitter, csm_correlation_threshold, pixels, nppp, num_repeats=num_repeats_simulation, z0=z0, detection_events_timestamps=detection_events_timestamps)

        simulation_data_dict[(nppp, pixels)] = {
            'ber_before': bit_error_ratios_before,
            'ber_after': bit_error_ratios_after,
            'ber_before_std': bit_error_ratios_before_std,
            'ber_after_std': bit_error_ratios_after_std
        }

        log = {
            'simulate_noise_peaks': simulate_noise_peaks,
            'simulate_lost_symbols': simulate_lost_symbols,
            'simulate_darkcounts': simulate_darkcounts,
            'simulate_jitter': simulate_jitter,
            'num_photons_per_pulse': num_photons_per_pulse,
            'darkcounts_factor': darkcounts_factor,
            'detector_jitter': detector_jitter,
            'detection_efficiencies': detection_efficiencies,
            'num_pixels': num_pixels,
            'data': simulation_data_dict
        }

        now = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
        with open(f'var_dump_simulate_decoding_with_file_intermediate_result_{now}', 'wb') as f:
            pickle.dump(log, f)


log = {
    'simulate_noise_peaks': simulate_noise_peaks,
    'simulate_lost_symbols': simulate_lost_symbols,
    'simulate_darkcounts': simulate_darkcounts,
    'simulate_jitter': simulate_jitter,
    'num_photons_per_pulse': num_photons_per_pulse,
    'darkcounts_factor': darkcounts_factor,
    'detector_jitter': detector_jitter,
    'detection_efficiencies': detection_efficiencies,
    'num_pixels': num_pixels,
    'data': simulation_data_dict
}

now = datetime.now().strftime('%d-%m-%Y--%H-%M-%S')
with open(f'var_dump_simulate_decoding_with_file_{now}', 'wb') as f:
    pickle.dump(log, f)


axs2: plt.Axes

fig, axs2 = plt.subplots(1)
axs2.errorbar(
    num_photons_per_pulse, bit_error_ratios_before[:, 0],
    bit_error_ratios_after_std[:, 0], 0,
    capsize=2,
    label='Before decoding',
    marker='o',
    markersize=5)

axs2.errorbar(
    num_photons_per_pulse, bit_error_ratios_after[:, 0],
    bit_error_ratios_before_std[:, 0], 0,
    capsize=2,
    label=f'After decoding ({num_photons_per_pulse[0]} photons per pulse)',
    marker='o',
    markersize=5)

axs2.errorbar(
    num_photons_per_pulse, bit_error_ratios_after[:, 1],
    bit_error_ratios_before_std[:, 1], 0,
    capsize=2,
    label=f'After decoding ({num_photons_per_pulse[1]} photons per pulse)',
    marker='o',
    markersize=5)


axs2.set_yscale('log')
axs2.set_ylabel('Bit Error Ratio (-)')
axs2.set_xlabel('Detection efficiency (%)')
plt.title('BER as function of detection efficiency')
plt.legend()
plt.show()

print('done')
print()


# %%

# input('done?')
