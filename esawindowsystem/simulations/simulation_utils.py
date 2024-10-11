import numpy.typing as npt
import numpy as np
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
        rng_gen=np.random.default_rng) -> npt.NDArray:
    """ Simulate the loss of symbols, based on the number of photons per pulse and detection efficiency.

    For each symbol, use the poisson distribution to determine how many photons arrived in each symbol pulse
    Then, do n bernoulli trials (binomial distribution) and success probability p, where n the number of photons
    per pulse and p is the detection efficiency. """

    num_symbols = len(peaks)

    num_photons_detected_per_pulse: npt.NDArray[np.int8] = rng_gen.binomial(
        rng_gen.poisson(num_photons_per_pulse, size=num_symbols),
        detection_efficiency)

    idxs_to_be_removed = np.where(num_photons_detected_per_pulse == 0)[0]
    print(f'Number of lost symbols: {len(idxs_to_be_removed):.0f}')
    print(f'Percentage lost: {len(idxs_to_be_removed)/peaks.shape[0]*100}')
    peaks = np.delete(peaks, idxs_to_be_removed)

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
        simulate_noise_peaks: bool,
        simulate_lost_symbols: bool,
        simulate_darkcounts: bool,
        darkcounts_fraction: float,
        simulate_jitter: bool,
        num_photons_per_pulse: int,
        detection_efficiency: float,
        detector_jitter: float,
        rng=np.random.default_rng(),
        rng_seed: int = 777):
    """Takes in the detection timestamps and simulates real world scenarios, such as loss of symbols, darkcounts and detection jitter. """
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

    timestamps = time_series[peaks]
    if simulate_darkcounts:
        darkcounts_timestamps = simulate_darkcounts_timestamps(darkcounts_fraction, peaks, rng_seed)
        timestamps = np.sort(np.hstack((timestamps, darkcounts_timestamps)))

    # timestamps = np.hstack((timestamps, rng.random(size=15) * timestamps[0]))
    # timestamps = np.sort(timestamps)

    if simulate_jitter:
        sigma = detector_jitter / 2.355
        timestamps += rng.normal(0, sigma, size=len(timestamps))

    peak_locations = timestamps
    peak_locations = np.sort(peak_locations)

    return peak_locations
