# %%
from fractions import Fraction
from math import ceil

import matplotlib.pyplot as plt
import numpy as np

# %% - Input parameters
N_bits = 80000              # Number of bits in the payload / image
M = 4                       # PPM order
N = 2                       # Number of parallel shift registers
CR = float(Fraction(2, 3))  # Code rate

num_samples_per_slot = 10                   # Number of DAC samples used for each slot
DAC_DATA_RATE = 8.82091E9                   # DAC data rate in Hz
sample_size_awg = 1 / DAC_DATA_RATE * 1E12  # Time duration of 1 DAC sample in ps
# Slot length in seconds
tau_slot = sample_size_awg * 1E-12 * num_samples_per_slot

ppm_orders = [4, 8, 16, 32, 64]
num_bits = [80000]

# %% - Calculate data rates for each parameter
data_rates = np.zeros((len(num_bits), len(ppm_orders)))

for ni, N_bits in enumerate(num_bits):
    for mi, M in enumerate(ppm_orders):
        m = np.log2(M)
        B = int(15120 / m / N)      # Base length of the shift register in the channel interleaver

        L_CSM = 24 if M == 4 else 16
        N_bits_IB = 15120 * CR - 2

        N_PPM_symbols_encoder = ceil(N_bits / N_bits_IB) * (N_bits_IB + 2) / (CR * np.log2(M))
        N_codewords = N_PPM_symbols_encoder / (15120 / np.log2(M))

        N_PPM_symbols = N_PPM_symbols_encoder + N_codewords * L_CSM + B * N * (N - 1)
        t_message = tau_slot * 5 / 4 * M * N_PPM_symbols

        datarate = N_bits / t_message * 1E-6
        data_rates[ni, mi] = datarate
        print(f'Datarate ({M} ppm): {datarate:.4f} Mbps')

# %% - Plot
ax = plt.axes()
for i in range(len(num_bits)):
    plt.semilogx(ppm_orders, data_rates[i], '-x', label=f'Nbits = {num_bits[i]/1000} kbits')
ax.set_xticks(ppm_orders)
ax.set_xticklabels([str(M) for M in ppm_orders])

plt.xlabel('PPM orders (-)')
plt.ylabel('Datarate (Mbps)')
plt.title('SCPPM data rates')
plt.legend()

# %%
