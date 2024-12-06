# %%
from math import ceil
from fractions import Fraction

import numpy as np

from esawindowsystem.core.encoder_functions import get_csm

CR = Fraction(2, 3)
M = 16
m = np.log2(M)
N_bits = 76000
CSM = get_csm(M)

slot_length = 1/8.82091E9*10

N = 4
B = int(15120/m/N)

N_bits_IB = 15120*CR - 2

N_PPM_symbols_encoder = ceil(N_bits/N_bits_IB)*(N_bits_IB+2)/(CR*m)

print('N', N_PPM_symbols_encoder)

N_codewords = N_PPM_symbols_encoder/(15120/m)

print('Codewords', N_codewords)

N_PPM_symbols = N_PPM_symbols_encoder + N_codewords*len(CSM) + B*N*(N-1)

print('PPM symbols', N_PPM_symbols)

t_message_seconds = slot_length*5/4*M*N_PPM_symbols
print('Message duration (ms)', t_message_seconds*1E3)

print('Symbols per second (Mcounts/s)', N_PPM_symbols/t_message_seconds*1E-6)
