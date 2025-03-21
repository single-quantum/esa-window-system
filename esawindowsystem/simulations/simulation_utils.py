import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import numpy.typing as npt

from esawindowsystem.core.utils import flatten


def print_parameter(parameter_str: str, parameter, spacing: int = 30):
    print(f'{parameter_str:<{spacing}} {parameter}')


def print_header(header: str, len_header: int = 50, filler: str = '-'):
    len_filler = (len_header - len(header)) // 2 - 2
    if len(header) % 2 == 0:
        print(f'{"#":{filler}<{len_filler}} {header} {"#":{filler}>{len_filler}}')
    else:
        print(f'{"#":{filler}<{len_filler}} {header} {"#":{filler}>{len_filler+1}}')


def simulate_symbol_loss(
        peaks: npt.NDArray,
        num_photons_per_pulse: int,
        detection_efficiency: float,
        num_pixels: int = 1,
        rng_gen=np.random.default_rng) -> npt.NDArray:
    """ Simulate the loss of symbols, based on the number of photons per pulse and detection efficiency.

    For each symbol, use the poisson distribution to determine how many photons arrived in each symbol pulse
    Then, do n bernoulli trials (binomial distribution) and success probability p, where n the number of photons
    per pulse and p is the detection efficiency. """

    num_symbols = len(peaks)

    # How many photons are present in a pulse
    num_photons_per_pulse_arr = rng_gen.poisson(num_photons_per_pulse, size=num_symbols)
    # num_photons_per_pulse = 4*np.ones(num_symbols, dtype=int)
    # Array that keeps track of how many detection events happened
    num_detections = np.zeros_like(num_photons_per_pulse_arr)

    total_number_of_photons = np.sum(num_photons_per_pulse_arr)
    absorbing_pixels = rng_gen.integers(0, num_pixels, total_number_of_photons)

    # not a correct name for this variable
    photons_absorbed = np.ones(absorbing_pixels.shape[0], dtype=int)

    j = 0
    for i, num_photons_in_pulse in enumerate(num_photons_per_pulse_arr):
        pixels = absorbing_pixels[j:j + num_photons_per_pulse_arr[i]]
        photons = photons_absorbed[j:j + num_photons_per_pulse_arr[i]]

        absorbed = np.zeros(num_pixels)
        for pi, pixel in enumerate(pixels):
            # If pixel did not yet absorb a photon
            if absorbed[pixel] == 0:
                photon_absorbed = rng_gen.binomial(photons[pi], detection_efficiency)
                if photon_absorbed == 1:
                    absorbed[pixel] += 1

        num_detections[i] = np.sum(absorbed)

        j += num_photons_per_pulse_arr[i]

    idxs_to_be_removed = np.where(num_detections == 0)[0]
    print(f'Number of lost symbols: {len(idxs_to_be_removed):.0f}')
    print(f'Percentage lost: {len(idxs_to_be_removed)/peaks.shape[0]*100}')

    num_detections[:4] = 1
    # Use the num detections array to make sure there is a timestamp for each detection event.
    # When `num_detections` has a 0 element, it is removed from the `peaks` array
    peaks = np.repeat(peaks, num_detections)

    return peaks


def simulate_darkcounts_timestamps(
        lmbda: float,
        msg_peaks: npt.NDArray[np.int_],
        time_series: npt.NDArray[np.float64],
        slot_length: float,
        rng_seed: int = 777):

    rng = np.random.default_rng(rng_seed)

    num_slots: int = int((time_series[msg_peaks][-1] - time_series[msg_peaks][0]) / slot_length)
    p: npt.NDArray[np.int_] = rng.poisson(lmbda, num_slots)

    darkcounts_timestamps: list[npt.NDArray[np.float64]] = []
    t0 = time_series[msg_peaks][0]

    darkcounts: npt.NDArray[np.float64]
    slot_start: float
    slot_end: float
    num_events: int

    for slot_idx, num_events in enumerate(p):
        if num_events == 0:
            continue
        slot_start = t0 + slot_idx * slot_length
        slot_end = t0 + (slot_idx + 1) * slot_length

        darkcounts = rng.uniform(slot_start, slot_end, num_events)
        darkcounts_timestamps.append(darkcounts)

    darkcounts_timestamps_flat: npt.NDArray[np.float64]
    darkcounts_timestamps_flat = np.array(flatten(darkcounts_timestamps))
    print(f'Inserted {darkcounts_timestamps_flat.shape[0]:.1e} darkcounts')

    return darkcounts_timestamps_flat


def get_simulated_message_peak_locations(
        msg_peaks: npt.NDArray[np.int_],
        time_series: npt.NDArray[np.float64],
        slot_length: float,
        simulate_noise_peaks: bool,
        simulate_lost_symbols: bool,
        simulate_darkcounts: bool,
        darkcounts_fraction: float,
        simulate_jitter: bool,
        num_photons_per_pulse: int,
        detection_efficiency: float,
        num_pixels: int,
        detector_jitter: float,
        rng=np.random.default_rng(),
        rng_seed: int = 777):
    """Takes in the detection timestamps and simulates real world scenarios, such as loss of symbols, darkcounts and detection jitter. """
    # Simulate noise peaks before start and after end of message
    peaks: npt.NDArray[np.int_]

    if simulate_lost_symbols:
        peaks = simulate_symbol_loss(msg_peaks, num_photons_per_pulse, detection_efficiency, num_pixels, rng_gen=rng)
    else:
        peaks = msg_peaks

    if simulate_noise_peaks:
        noise_peaks: npt.NDArray[np.int_] = np.sort(rng.integers(0, msg_peaks[0], 1))
        # noise_peaks = np.array([53])
        noise_peaks_end = np.sort(rng.integers(peaks[-1], len(time_series), 15))
        peaks = np.hstack((noise_peaks, peaks, noise_peaks_end))

    timestamps = time_series[peaks]
    if simulate_darkcounts:
        darkcounts_timestamps = simulate_darkcounts_timestamps(
            darkcounts_fraction, peaks, time_series, slot_length, rng_seed)
        timestamps = np.sort(np.hstack((timestamps, darkcounts_timestamps)))

    # timestamps = np.hstack((timestamps, rng.random(size=15) * timestamps[0]))
    # timestamps = np.sort(timestamps)

    if simulate_jitter:
        sigma = detector_jitter / 2.355
        timestamps += rng.normal(0, sigma, size=len(timestamps))

    peak_locations = timestamps
    peak_locations = np.sort(peak_locations)

    return peak_locations
