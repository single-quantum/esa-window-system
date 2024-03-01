# This file shows the effect of different sigmas (AWGN) on the BPSK modulation scheme.

import matplotlib.pyplot as plt
from numpy.random import default_rng


def bpsk_encoding(input_sequence, sigma=0.8):
    """Use BPSK to modulate the bit array. """
    rng = default_rng()

    output_1 = []
    output_2 = []

    for ri in input_sequence:
        if ri == 0:
            output_1.append(-1 + rng.normal(0, sigma))
        else:
            output_2.append(1 + rng.normal(0, sigma))

    return output_1, output_2


rng = default_rng()
signal = rng.integers(0, 2, 10000)

sigmas = [0.1, 0.5, 1]

fig, axs = plt.subplots(1, 3, sharey=True)
ylims = []
for i, sigma in enumerate(sigmas):
    output_1, output_2 = bpsk_encoding(signal, sigma)
    hist = axs[i].hist(output_1, bins=50, label='Bit value: 0')
    axs[i].hist(output_2, bins=50, label='Bit value: 1')

    ylims.append(max(hist[0]))

    axs[i].set_xlim(-4, 4)
    axs[i].set_xlabel('BPSK value (AWGN)')
    axs[i].text(2.5, 5, rf'$\sigma$={sigmas[i]}')

ylim = max(ylims)
for i in range(len(sigmas)):
    axs[i].vlines(-1, 0, ylim, color='r', linestyles='dashed')
    axs[i].vlines(1, 0, ylim, color='r', linestyles='dashed')
    axs[i].set_ylim(0, ylim)

axs[1].legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", mode="expand", ncol=2)
axs[0].set_ylabel('Count')
fig.suptitle('BPSK mapping with AWGN')
plt.show()
