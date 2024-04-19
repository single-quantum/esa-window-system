from fractions import Fraction

import numpy.typing as npt
from numpy import log2, int_
from PIL import Image

from esawindowsystem.core.encoder_functions import get_csm

num_samples_per_slot: int = 10          # Number of DAC samples in one slot
M: int = 16                             # each m = 4 bits are mapped from 0 to M = 16
CODE_RATE = Fraction(2, 3)

PAYLOAD_TYPE: str = 'image'
IMG_FILE_PATH = "esawindowsystem/sample_payloads/herbig-haro-211-small.png"
img = Image.open(IMG_FILE_PATH)
IMG_SHAPE: tuple[int, int] = (img.size[1], img.size[0])
GREYSCALE: bool = True
USE_INNER_ENCODER = True
USE_RANDOMIZER = True

# To calculate where the codewords are, a correlation is made with the CSM. The threshold,
# as percentage of the maximum correlation of the received sequence, 'decides' whether a peak
# belongs to the start of a codeword or not
CORRELATION_THRESHOLD: float = 0.50
DEBUG_MODE: bool = False

# Usually we record multiple copies of the same message. You can use this to select a particular copy of the message,
# as sometimes a certain copy of the message has too much noise to recover.
MESSAGE_IDX: list[int] = [0, 1]

# No need to change the parameters below, unless you know what you're doing!
# Note: The fraction type is needed for proper match casing
DAC_DATA_RATE: float = 8.82091E9        # DAC data rate in Hz

# PPM parameters
m: int = int(log2(M))
slot_factor: float = 5 / 4  # 1 means: no guard slot, 5/4 means: M/4 guard slots


# Channel interleaver parameters
# The length of each shift register is B*N, with N going from 0 to N-1
CHANNEL_INTERLEAVE: bool = True
N_interleaver: int = 2               # The number of parallel shift registers

num_symbols_per_slice: int = 15120
B_interleaver = int((num_symbols_per_slice / m) / 2)            # The base length of each shift register
symbols_per_codeword: int = num_symbols_per_slice // m

if CHANNEL_INTERLEAVE:
    if not (B_interleaver * N_interleaver) % symbols_per_codeword == 0:
        raise ValueError("The product of B and N should be a multiple of 15120/m")

BIT_INTERLEAVE: bool = True

num_slots_per_symbol: int = int(slot_factor * M)

CSM: npt.NDArray[int_] = get_csm(M)

sample_size_awg: float = 1 / DAC_DATA_RATE * 1E12       # Time duration of 1 DAC sample in ps
slot_length: float = sample_size_awg * 1E-12 * num_samples_per_slot  # Length of 1 bin in time
symbol_length: float = slot_length * num_slots_per_symbol          # Length of 1 symbol in time
