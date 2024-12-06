# %%
"""This script is used to acquire time tags from the Swabian time tagger ultra. """
from datetime import datetime
from time import sleep
from pathlib import Path
import pickle
import numpy as np
import numpy.typing as npt
from scipy.constants import h, c
import scipy as sp

import TimeTagger
import matplotlib.pyplot as plt

# from esawindowsystem.ppm_parameters import CODE_RATE, M, num_samples_per_slot, IMG_FILE_PATH, GREYSCALE, slot_length, symbol_length, PAYLOAD_TYPE, IMG_SHAPE, USE_INNER_ENCODER, USE_RANDOMIZER


# %%

tagger = TimeTagger.createTimeTagger(resolution=TimeTagger.Resolution.Standard)
channels = [1, 2, 3, 4]

for i in channels:
    tagger.setDeadtime(i, 100)
    tagger.setTriggerLevel(i, 0.50)


binwidth = 2        # Bin width in ps
num_bins = 700     # Number of bins
integration_time_seconds = 30
detector = 'C824-18'

reference_channel = 1
for correlation_channel in [2, 3, 4]:
    corr = TimeTagger.Correlation(tagger, correlation_channel, reference_channel, binwidth=binwidth, n_bins=num_bins)
    # corr = TimeTagger.StartStop(tagger, 4, 5, binwidth=binwidth)

    # Run Correlation for 1 second to accumulate the data
    corr.startFor(int(integration_time_seconds*1e12), clear=True)
    corr.waitUntilFinished()

    # Read the correlation data
    data: npt.NDArray[np.float64] = corr.getData()
    x_axis = corr.getIndex()

    plt.figure()
    plt.plot(x_axis, data)
    plt.show()

    time_tagger_data = {
        'histogram_bins': x_axis,
        'histogram_bin_values': data,
        'reference_channel': reference_channel,
        'correlation_channel': correlation_channel,
        'bin_width': binwidth,
        'num_bins': num_bins,
        'integration_time_seconds': integration_time_seconds
    }

    with open(f'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\C824-18\\correlation_measurement_{detector}_reference-{reference_channel}_correlation-channel-{correlation_channel}', 'wb') as f:
        pickle.dump(time_tagger_data, f)

# %%
detector = 'C824-02'
reference_channel = 1
for correlation_channel in [2, 3, 4]:
    with open(f'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\{detector}\\correlation_measurement_{detector}_reference-{reference_channel}_correlation-channel-{correlation_channel}', 'rb') as f:
        time_tagger_data = pickle.load(f)

    histogram_bins = time_tagger_data['histogram_bins']
    histogram_bin_values = time_tagger_data['histogram_bin_values']

    plt.figure()
    plt.title(f'Cross correlation between channel {reference_channel} and channel {reference_channel}')
    plt.plot(histogram_bins, histogram_bin_values)
    plt.xlabel('Time (ps)')
    plt.ylabel('Occurences (-)')
    plt.show()

    where_max_bin = np.argmax(histogram_bin_values)
    relative_time_delay = histogram_bins[where_max_bin]

    print(f'Time delay between channel {reference_channel} and {correlation_channel}: {relative_time_delay:.2f} ps')

# %%

reference_channel = 1
correlation_channel = 3

with open(f'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\{detector}\\expla-source-correlation-pixel-{reference_channel}-start-pixel-{correlation_channel}-stop.txt', 'rb') as f:
    time_tagger_data = np.loadtxt(f.name, delimiter='\t', dtype=int, skiprows=1)

histogram_bins = time_tagger_data[:, 0]
histogram_bin_values = time_tagger_data[:, 1]

plt.figure()
plt.title(f'Cross correlation between channel {reference_channel} and channel {correlation_channel}')
plt.plot(histogram_bins, histogram_bin_values)
plt.xlabel('Time (ps)')
plt.ylabel('Occurences (-)')
plt.show()


# %%


def calc_SNR(y: npt.NDArray[np.float64]):
    peaks = sp.signal.find_peaks(y, height=15000)
    peak_indeces = peaks[0]

    SNRs: list[float] = []

    for i in range(len(peak_indeces) - 1):
        N: int = peak_indeces[i]
        N_noise: int = peak_indeces[i] + int((peak_indeces[i+1]-peak_indeces[i])/2)

        half_peak_width: int = 7
        signal: float = np.sum(y[N-half_peak_width:N+half_peak_width])/(2*half_peak_width)
        noise: float = np.sum(y[N_noise-half_peak_width:N_noise+half_peak_width])/(2*half_peak_width)

        SNRs.append(signal/noise)

    return np.mean(SNRs)


def calc_num_photons_per_pulse(
    num_pulses_per_second: float,
    reference_power: float,
    lmbda: float = 1550E-9
) -> float:
    attenuation_to_output: float = 21  # Attenuation between reference and output in dB
    detector_efficiency: float = 0.5

    photon_energy: float = h*c/lmbda   # Photon energy in Joule

    if isinstance(reference_power, str):
        reference_power = float(reference_power)

    reference_power = reference_power*1E-9

    output_power: float = reference_power*10**(-attenuation_to_output/10)/detector_efficiency

    num_photons_per_second: float = output_power/photon_energy
    num_photons_per_pulse: float = num_photons_per_second / num_pulses_per_second

    return num_photons_per_pulse


# num_channels = 4
# channels = [i+1 for i in range(num_channels)]
channels = [1, 2, 3, 4]

tagger = TimeTagger.createTimeTagger(resolution=TimeTagger.Resolution.Standard)
# sampling_time_ps = tagger.getPsPerClock()
# print(f'{1/(sampling_time_ps*1E-12):.3e}')
# print('used clock', tagger.xtra_getClockSource())
serial = tagger.getSerial()
print(f'Connected to time tagger {serial}')

for i in channels:
    tagger.setDeadtime(i, 100)
    tagger.setTriggerLevel(i, 0.15)

tagger.setTriggerLevel(5, 0.15)
tagger.sync()

sleep(1)

current_time = datetime.now()
timestamp_epoch = int(datetime.timestamp(datetime.now()))
print('Current time: ', current_time)
formatted_time = current_time.strftime("%H-%M-%S")
window_size_secs: float = 16E-3
window_size_ps: float = window_size_secs * 1E12  # Window time in ps

cr = str(CODE_RATE).replace('/', '-')

img_path = Path(IMG_FILE_PATH)
img_name = img_path.name.rstrip(img_path.suffix)

# filewriter = TimeTagger.FileWriter(
#     tagger,
#     f'time tagger files/{img_name}_{num_samples_per_slot}-sps_{M}-PPM_{cr}-code-rate_{formatted_time}_{timestamp_epoch}',
#     channels=channels)

# filewriter.startFor(int(window_size_ps), clear=True)
# filewriter.waitUntilFinished()

binwidth = 10       # Bin width in ps
num_bins = 10000     # Number of bins

corr = TimeTagger.Correlation(tagger, 4, 5, binwidth=binwidth, n_bins=num_bins)
# corr = TimeTagger.StartStop(tagger, 4, 5, binwidth=binwidth)

# Run Correlation for 1 second to accumulate the data
corr.startFor(int(1*1e12), clear=True)
corr.waitUntilFinished()

# Read the correlation data
data: npt.NDArray[np.float64] = corr.getData()
x_axis = corr.getIndex()

filewriter = TimeTagger.FileWriter(
    tagger,
    f'time tagger files/{img_name}_{num_samples_per_slot}-sps_{M}-PPM_{cr}-code-rate_{formatted_time}_{timestamp_epoch}',
    channels=channels)

filewriter.startFor(int(window_size_ps), clear=True)
filewriter.waitUntilFinished()

num_events: int = filewriter.getTotalEvents()
events_per_second: float = num_events/window_size_secs

# reference_power = input("Reference power (nW): ")

SNR = calc_SNR(data)
# num_photons_per_pulse = calc_num_photons_per_pulse(events_per_second, reference_power)

print(f'{num_events} events written to disk. ')
print(f'Events per second: {events_per_second:.3e}')
# print(f'Number of photons per second: {num_photons_per_pulse}')
print('SNR:', SNR)

# plt.figure()
# plt.plot(x_axis, data)
# plt.title('DAC / SNSPD correlation')
# plt.xlabel('Shift (ps)')
# plt.ylabel('Occurences (-)')
# plt.show()

# with open(f'DAC_SNSPD_correlation_calibration_message_{M}_ppm_{num_samples_per_slot}sps_1ns_pulse_width_gaussian', 'wb') as f:
#     pickle.dump({'correlation_data': data, 'bin_times': x_axis}, f)

# %%

filewriter = TimeTagger.FileWriter(
    tagger,
    f'time tagger files/efficiency_recovery_test',
    channels=channels)

filewriter.startFor(10*1E12, clear=True)
filewriter.waitUntilFinished()

fr = TimeTagger.FileReader('time tagger files/efficiency_recovery_test')
data = fr.getData(num_events)
time_stamps = data.getTimestamps()

diff_time_stamps = np.diff(time_stamps)

plt.figure()
plt.hist(diff_time_stamps, bins=100, cumulative=True)
plt.show()

print('len', len(time_stamps))


# %%

slot_widths = np.array([20, 10, 5, 3])*0.1133

plt.figure()
plt.plot(slot_widths, [2, 2, 1, 1], marker='o', linestyle='None', label='Without bridges', markersize=10)
plt.plot(slot_widths, [2, 2, 2, 2], marker='x', linestyle='None', label='With bridges', markersize=10)
plt.xticks(slot_widths, [f'{s:.2f}' for s in slot_widths], fontsize=16)
plt.xlabel('Slot width (ns)', fontsize=16)
plt.yticks([1, 2], ['Not decodeable', 'Decodeable'], fontsize=16)
plt.title('Decoding with fast / normal SNSPDs', fontsize=20)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()


# %%
A = np.genfromtxt("C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Waveforms\\Detector waveforms\\RefCurve_2024-11-01_1_162753.Wfm.csv", delimiter=",")
t = np.linspace(-5.035E-8, 4.964E-8, 4000)
plt.figure()
plt.plot(t*1E9, A)
plt.xlabel('Time (ns)')
plt.show()


# %%

with open('C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\time tagger files\\timetags_metadata_1732094455', 'rb') as f:
    file_data = pickle.load(f)

plt.figure()
plt.plot(file_data['dac_snspd_correlation_bin_times'][:2000], file_data['dac_snspd_correlation_histogram'][:2000])
plt.show()
