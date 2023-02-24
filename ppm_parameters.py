from fractions import Fraction

from numpy import log2

from encoder_functions import get_csm

DAC_DATA_RATE: float = 8.82091E9       # DAC data rate in Hz
num_samples_per_slot: int = 8            # Number of samples per bin

# Note: The fraction type is needed for proper match casing
CODE_RATE = Fraction(2, 3)
PAYLOAD_TYPE: str = 'image'
GREYSCALE: bool = True
IMG_SHAPE: tuple[int, int] = (95, 100)

# PPM parameters
M: int = 8                  # each m = 4 bits are mapped from 0 to M = 16
m: int = int(log2(M))
slot_factor: float = 5 / 4   # 1 means: no guard slot, 5/4 means: M/4 guard slots


# Channel interleaver parameters
# The length of each shift register is B*N, with N going from 0 to N-1
CHANNEL_INTERLEAVE = True
B_interleaver = 2520            # The base length of each shift register
N_interleaver = 2               # The number of parallel shift registers

num_symbols_per_slice: int = 15120
symbols_per_codeword = num_symbols_per_slice // m

if CHANNEL_INTERLEAVE:
    if not (B_interleaver * N_interleaver) % symbols_per_codeword == 0:
        raise ValueError("The product of B and N should be a multiple of 15120/m")

BIT_INTERLEAVE = True


num_bins_per_symbol: int = int(slot_factor * M)

CSM = get_csm(M)

sample_size_awg: float = 1 / DAC_DATA_RATE * 1E12       # Time duration of 1 DAC sample in ps
bin_length: float = sample_size_awg * 1E-12 * num_samples_per_slot  # Length of 1 bin in time
symbol_length: float = bin_length * num_bins_per_symbol          # Length of 1 symbol in time
