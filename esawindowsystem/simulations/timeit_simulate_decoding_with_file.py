# %%
from timeit import timeit

setup_code = '''
import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
# import TimeTagger
from numpy.random import default_rng
from PIL import Image
from scipy.signal import find_peaks

from esawindowsystem.core.BCJR_decoder_functions import \
    ppm_symbols_to_bit_array
from esawindowsystem.core.demodulation_functions import demodulate
from esawindowsystem.core.encoder_functions import map_PPM_symbols
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


                                            
def print_parameter(parameter_str: str, parameter, spacing: int = 30):
    print(f'{parameter_str:<{spacing}} {parameter}')


def print_header(header: str, len_header: int = 50, filler='-'):
    len_filler = (len_header - len(header)) // 2 - 2
    if len(header) % 2 == 0:
        print(f'{"#":{filler}<{len_filler}} {header} {"#":{filler}>{len_filler}}')
    else:
        print(f'{"#":{filler}<{len_filler}} {header} {"#":{filler}>{len_filler+1}}')


def simulate_symbol_loss(
        peaks: npt.NDArray,
        num_photons_per_pulse: int,
        detection_efficiency: float,
        rng_gen=np.random.default_rng) -> npt.NDArray:
    """ Simulate the loss of symbols, based on the number of photons per pulse and detection efficiency.

    For each symbol, use the poisson distribution to determine how many photons arrived in each symbol pulse
    Then, do n bernoulli trials (binomial distribution) and success probability p, where n the number of photons
    per pulse and p is the detection efficiency. """

    num_symbols = len(peaks)

    num_photons_detected_per_pulse = rng_gen.binomial(
        rng_gen.poisson(num_photons_per_pulse, size=num_symbols),
        detection_efficiency)

    idxs_to_be_removed = np.where(num_photons_detected_per_pulse == 0)[0]
    print(f'Number of lost symbols: {len(idxs_to_be_removed):.0f}')
    print(f'Percentage lost: {len(idxs_to_be_removed)/peaks.shape[0]*100}')
    peaks = np.delete(peaks, idxs_to_be_removed)

    return peaks


def simulate_darkcounts_timestamps(rng, lmbda: float, msg_peaks: npt.NDArray[np.int_]):
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


def get_simulated_message_peak_locations(
        msg_peaks: npt.NDArray[np.int_],
        time_series: npt.NDArray[np.float64],
        simulate_noise_peaks: bool,
        simulate_lost_symbols: bool,
        simulate_darkcounts: bool,
        simulate_jitter: bool,
        rng=np.random.default_rng()):
    # Simulate noise peaks before start and after end of message
    peaks: npt.NDArray[np.int_]

    if simulate_noise_peaks:
        noise_peaks: npt.NDArray[np.int_] = np.sort(rng.integers(0, msg_peaks[0], 15))
        noise_peaks[0] += 1
        noise_peaks_end = np.sort(rng.integers(msg_peaks[-1], len(time_series), 15))
        peaks = np.hstack((noise_peaks, msg_peaks, noise_peaks_end))
    else:
        peaks = msg_peaks

    if simulate_lost_symbols:
        peaks = simulate_symbol_loss(peaks, num_photons_per_pulse, detection_efficiency, rng_gen=rng)

    num_symbols_received = len(peaks)

    timestamps = time_series[peaks]
    if simulate_darkcounts:
        darkcounts_timestamps = simulate_darkcounts_timestamps(rng, 0.01, peaks)
        timestamps = np.sort(np.hstack((timestamps, darkcounts_timestamps)))

    # timestamps = np.hstack((timestamps, rng.random(size=15) * timestamps[0]))
    # timestamps = np.sort(timestamps)

    if simulate_jitter:
        sigma = detector_jitter / 2.355
        timestamps += rng.normal(0, sigma, size=len(timestamps))

    if simulate_darkcounts and num_darkcounts > 0:
        SNR = 10 * np.log10(num_symbols_received / num_darkcounts)
        print('Signal: ', num_symbols_received, 'Noise: ', num_darkcounts, 'SNR: ', SNR)

    peak_locations = timestamps
    peak_locations = np.sort(peak_locations)

    return peak_locations


simulate_noise_peaks: bool = True
simulate_lost_symbols: bool = False
simulate_darkcounts: bool = False
simulate_jitter: bool = False

detection_efficiency_lower: float = 0.7
detection_efficiency_upper: float = 0.8
detection_efficiency_step_size = 0.1
num_photons_per_pulse = 5
darkcounts_factor: float = 0.05
detector_jitter = 50E-12

use_test_file: bool = True
use_latest_tt_file: bool = False
compare_with_original: bool = False
plot_BER_distribution: bool = False
num_samples_ppm_pulse = 10

time_events_filename: Path
base_dir = Path.cwd() / Path('esawindowsystem') / Path('ppm_sample_messages')
reference_file_path = Path.cwd(
) / f'herbig_haro_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence'

if use_test_file:
    cr = str(CODE_RATE).replace('/', '-')

    if GREYSCALE:
        time_events_filename = base_dir / Path(f'ppm_message_SQ_tiny_greyscale_{IMG_SHAPE[0]}x{IMG_SHAPE[1]}_pixels_{M}' +
                                               f'-PPM_{num_samples_per_slot}_{num_samples_ppm_pulse}_c1b1_{cr}-code-rate.csv')
    else:
        time_events_filename = base_dir / Path(f'ppm_message_SQ_tiny_{IMG_SHAPE[0]}x{IMG_SHAPE[1]}_pixels_{M}' +
                                               f'-PPM_{num_samples_per_slot}_{num_samples_ppm_pulse}_c1b1_{cr}-code-rate.csv')

elif not use_test_file and use_latest_tt_file:
    tt_files_dir = 'time tagger files/'
    tt_files_path = Path(__file__).parent.absolute() / tt_files_dir
    tt_files = tt_files_path.rglob('*.ttbin')
    files: list[Path] = [x for x in tt_files if x.is_file()]
    files = sorted(files, key=lambda x: x.lstat().st_mtime)
    time_events_filename = tt_files_dir + re.split(r'\.\d{1}', files[-1].stem)[0] + '.ttbin'
else:
    time_events_filename = 'time tagger files/jupiter_tiny_greyscale' + \
        '_16_samples_per_slot_CSM_0_interleaved_15-56-53.ttbin'

slot_width_ns = num_samples_per_slot * sample_size_awg / 1000
symbol_length_ns = num_slots_per_symbol * slot_width_ns


msg_peaks: npt.NDArray[np.int_] = np.array([])
time_series: npt.NDArray[np.float64] = np.array([])
peak_locations: npt.NDArray[np.float64] = np.array([])

if use_test_file:
    if not Path.is_file(Path(time_events_filename)):
        generate_awg_pattern(num_samples_ppm_pulse)
        time_events_filename = Path('esawindowsystem') / Path('ppm_sample_messages') / Path(time_events_filename)
    samples = pd.read_csv(time_events_filename, header=None)
    print(f'Decoding file: {time_events_filename}')
    samples_np_array: npt.NDArray[np.int_] = samples.to_numpy().flatten()

    # Make a time series based on the length of samples and how long one sample is in time
    time_series_end = len(samples_np_array) * sample_size_awg * 1E-12
    time_series = np.arange(0, time_series_end, sample_size_awg * 1E-12)

    msg_peaks = find_peaks(samples_np_array, height=1, distance=2)[0]



cached_trellis: Trellis | None = None

cached_trellis_file_path = Path('cached_trellis_80640_timesteps')


z = 0

# SEED = 21189 + z**2
SEED = 21190 + z**2
print('Seed', SEED, 'z', z)
rng = default_rng(SEED)

num_darkcounts: int = 0
if use_test_file:
    peak_locations = get_simulated_message_peak_locations(
        msg_peaks, time_series, simulate_noise_peaks, simulate_lost_symbols, simulate_darkcounts, simulate_jitter, rng)
'''

execution_code = '''
slot_mapped_message, _, _ = demodulate(
    peak_locations[:200000], M, slot_length, symbol_length, csm_correlation_threshold=0.75, **{'debug_mode': False})


information_blocks: npt.NDArray[np.int_] = np.array([])


information_blocks, BER_before_decoding = decode(
    slot_mapped_message, M, CODE_RATE, CHANNEL_INTERLEAVE=True, BIT_INTERLEAVE=True, use_inner_encoder=True,
    **{
        'use_cached_trellis': False,
        # 'cached_trellis_file_path': cached_trellis_file_path,
        'user_settings': {'reference_file_path': reference_file_path},
    })
'''

num_tries = 5
execution_time = timeit(setup=setup_code, stmt=execution_code, number=num_tries)
print('Execution time: ', execution_time/num_tries)

# %%
