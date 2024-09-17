import math
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt

from esawindowsystem.core.shift_register import CRC

BitArray = npt.NDArray[np.int_]


def validate_PPM_order(M: int):
    if M not in (4, 8, 16, 32, 64, 128, 256):
        raise ValueError("M should be one of 4, 8, 16, 32, 64, 128 or 256")


def get_asm_bit_arr(asm_hex: str = '1ACFFC1D') -> BitArray:
    """Returns the binary representation of the Attached Synchronisation Markerk, as defined by the CCSDS. """
    ASM_binary = np.binary_repr(int(asm_hex, base=16), width=32)
    ASM_arr = np.array([int(i) for i in ASM_binary])

    return ASM_arr.astype(int)


def prepend_asm(arr: BitArray) -> BitArray:
    """Prepend `arr` with the Attached Synchronization Marker. """
    ASM_arr = get_asm_bit_arr()
    # Tile the ASM, such that it vertically, so that it can be stacked to the reshaped array
    asm_bit_array = np.tile(ASM_arr, arr.shape[0]).reshape(arr.shape[0], -1)

    return np.hstack((asm_bit_array, arr))


def generate_pseudo_randomized_sequence(seed: list[int] = [1] * 8) -> list[int]:
    """Generate the pseudo randomized sequence, as defined in the CCSDS standard. """
    sequence: list[int] = []

    sequence.append(seed[-1])

    addition_1: int = seed[4] ^ seed[7]
    addition_2: int = seed[2] ^ addition_1
    addition_3: int = seed[0] ^ addition_2

    output = np.roll(seed, 1).tolist()
    output[0] = addition_3

    sequence.append(output[-1])
    i = 2

    while i < 256:
        # print('input', output)
        addition_1 = output[4] ^ output[7]
        addition_2 = output[2] ^ addition_1
        addition_3 = output[0] ^ addition_2

        output = np.roll(output, 1).tolist()
        output[0] = addition_3

        sequence.append(output[-1])

        i += 1

    return sequence


def randomize(information_blocks: BitArray) -> BitArray:
    """Pseudo randomize the information blocks, using the shift register defined by the CCSDS.

    Note that performing the randomize function twice gives back the de-randomized sequence. """
    initial_shape = information_blocks.shape
    pseudo_randomized_sequence = generate_pseudo_randomized_sequence()

    output_array = np.zeros(initial_shape, dtype=int)
    # information_blocks = information_blocks.flatten()

    if information_blocks.ndim == 1:
        copies = math.ceil(len(information_blocks) / len(pseudo_randomized_sequence))
        pseudo_randomized_sequence = np.tile(pseudo_randomized_sequence, copies)
        output_array = information_blocks ^ pseudo_randomized_sequence[:len(information_blocks)]
        return output_array

    i = 0
    for row in information_blocks:
        # Tile the randomized sequence, so that it is long enough to XOR with the information blocks
        copies = math.ceil(len(row) / len(pseudo_randomized_sequence))
        pseudo_randomized_sequence = np.tile(pseudo_randomized_sequence, copies)

        # XOR addition of both sequences.
        randomized_row = row ^ pseudo_randomized_sequence[:len(row)]
        output_array[i] = randomized_row
        i += 1

    return output_array


def get_CRC(arr: BitArray) -> BitArray:
    CRC_size: int = 32

    # Initialize the CRC shift register
    seed: list[int] = [1] * CRC_size
    sr = CRC(seed, [3, 14, 18, 29])

    for j in range(arr.shape[0] - CRC_size):
        sr.next(arr[j] ^ sr.state[-1])

    return sr.state


def append_CRC(arr: BitArray):
    # Fill the input array `arr` with 32 zeros, so that the CRC can be attached
    CRC_size = 32
    arr = np.concatenate((arr, np.zeros((arr.shape[0], CRC_size), dtype=arr.dtype)), axis=1)

    for i in range(arr.shape[0]):
        # Attach to the arr
        arr[i, -CRC_size:] = get_CRC(arr[i, :])

    return arr


def slicer(arr: BitArray, code_rate: Fraction, include_crc: bool = False,
           len_CRC: int = 32, num_termination_bits: int = 2) -> BitArray:
    """Slice the input array into information blocks, based on the code rate. """

    # For a code rate of 1/3, the information block size is 5006.
    # Including CRC and 2 termination bits, the number of bits going into the encoder is 5040
    information_block_size: int = int(15120 * float(code_rate) - len_CRC - num_termination_bits)

    # If the CRC is not used, add 32 more bits to the information block.
    if not include_crc:
        information_block_size += len_CRC

    if arr.shape[0] % information_block_size != 0:
        # Number of bits that need to be added such that the array is a multiple of the information block size
        num_pad_bits = information_block_size - arr.shape[0] % information_block_size
        arr = np.hstack((arr, np.zeros(num_pad_bits, dtype=int)))

    arr = arr.reshape((-1, information_block_size))
    return arr


def unpuncture(encoded_sequence: BitArray, code_rate: Fraction,
               dtype: type[int] | type[float] = int) -> BitArray:
    puncture_scheme: dict[Fraction, list[int]] = {
        Fraction(1, 3): [1, 1, 1, 1, 1, 1],
        Fraction(1, 2): [1, 1, 0, 1, 1, 0],
        Fraction(2, 3): [1, 1, 0, 0, 1, 0]
    }

    P = puncture_scheme[code_rate]

    factor = code_rate / Fraction(1, 3)
    unpunctured_sequence = np.zeros(int(factor * len(encoded_sequence)), dtype=dtype)

    j = 0
    for i in range(len(unpunctured_sequence) - 1):
        if P[i % len(P)] == 1:
            unpunctured_sequence[i] = encoded_sequence[j]
            j += 1

    return unpunctured_sequence


def puncture(convoluted_bit_sequence: npt.NDArray[np.int_ | np.float64],
             code_rate: Fraction, dtype: type[int] | type[float] = int) -> BitArray:
    """If the code rate is not 1/3, puncture (remove) elements according to the scheme defined by the CCSDS. """
    puncture_scheme: dict[Fraction, list[int]] = {
        Fraction(1, 3): [1, 1, 1, 1, 1, 1],
        Fraction(1, 2): [1, 1, 0, 1, 1, 0],
        Fraction(2, 3): [1, 1, 0, 0, 1, 0]
    }

    # THE CCSDS HATH SPOKEN:
    # "3.8.2.3.2 The puncturing shall be accomplished using the following procedure:"
    # (See page 3-12 of the CCSDS 142.0-B-1 blue book, August 2019 edition)

    convolutional_codewords: BitArray = np.zeros(
        (convoluted_bit_sequence.shape[0], int(convoluted_bit_sequence.shape[1] / (3 * float(code_rate)))), dtype=dtype
    )
    P = puncture_scheme[code_rate]
    for i, row in enumerate(convoluted_bit_sequence):
        j = 0
        for m in range(row.shape[0]):
            if P[m % 6] == 1:
                convolutional_codewords[i, j] = convoluted_bit_sequence[i, m]
                j += 1

    return convolutional_codewords


def zero_terminate(arr: BitArray, num_termination_bits: int = 2) -> BitArray:
    zero_bits = np.zeros((arr.shape[0], num_termination_bits), dtype=int)
    return np.hstack((arr, zero_bits))


def accumulate(arr: BitArray) -> BitArray:
    """Accumulate XOR-wise `arr` with itself. """
    n_j = np.zeros_like(arr)
    n_j[0] = arr[0]
    for j in range(1, n_j.shape[0]):
        n_j[j] = n_j[j - 1] ^ arr[j]

    return n_j


def convolve(
        arr: BitArray | tuple[int, ...],
        initial_state: tuple[int, int] = (0, 0)) -> tuple[BitArray, tuple[int, ...]]:
    """Use a convolutional shift register to generate a convoluted codeword.

    For more details on the convolutional encoder see CCSDS blue book 142.0-B-1, section 3.8.2 (August 2019)"""
    # Number of sliding windows that are iterated over to generate the
    # convolutional codeword

    num_windows: int = arr.shape[0] if isinstance(arr, np.ndarray) else len(arr)
    convolutional_codeword: BitArray = np.zeros(3 * num_windows, dtype=int)

    # To initialize `arr`, add the initial state.
    arr = np.hstack((tuple(reversed(initial_state)), arr))
    f: BitArray = np.array([])

    for i in range(num_windows):
        f = arr[i:i + 3]

        h = [f[2] ^ f[0], f[0] ^ f[1] ^ f[2], f[0] ^ f[1] ^ f[2]]
        convolutional_codeword[3 * i:3 * i + 3] = h

    terminal_state: tuple[int, ...] = tuple(np.flip(f[1:]))

    return convolutional_codeword, terminal_state


def map_PPM_symbols(arr: list[int] | tuple[int, ...] | BitArray, m: int):
    """Map input array of bits to PPM symbols. """
    # Input validation
    validate_PPM_order(2**m)

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=int)

    # Check if PPM symbols are consistent with the PPM order
    if not np.all(arr < 2**m):
        raise ValueError(f"All PPM symbols should be smaller than 2^{m}")

    # Check if array is a bit array
    if not np.all(np.logical_or(arr == 0, arr == 1)):
        raise ValueError("Array is not a bit array. All inputs should be 1 or 0. ")

    if arr.shape[0] // m != arr.shape[0] / m:
        raise ValueError(f"Input array is not a multiple of m={m}")

    S = arr.shape[0] // m

    output_arr = np.zeros(S, dtype=int)

    for s in range(S):
        q = np.zeros(m)
        for a in range(m):
            q[a] = 2**(m - a - 1) * arr[m * s + a]
        output_arr[s] = np.sum(q)

    return output_arr


def bit_interleave(arr: BitArray, dtype: type[int] | type[float] = int) -> BitArray:
    """Shuffle some bits around to make a so-called bit-interleaved codeword.

    Note: works only with 15120 element arrays. The modulo 15120 is hard coded for a reason.
    This is because the permutation polynomial in `bit_deinterleave` only works properly with 15120 element arrays.
    For any other length array, the interleaved array might not be invertable. """

    if arr.shape[0] != 15120:
        raise ValueError("Input array should have length 15120")

    interleaved_output = np.zeros_like(arr).astype(dtype)

    # pi_j is the bit index from the original bit array
    for j in range(interleaved_output.shape[0]):
        pi_j = (11 * j + 210 * j**2) % 15120
        interleaved_output[j] = arr[pi_j]

    return interleaved_output


def bit_deinterleave(arr: BitArray | list[float], dtype: Any = int) -> BitArray:
    """De-interleave the interleaved array `arr`.

    Note: works only with 15120 element arrays. """

    assert len(arr) == 15120, "Input array should have length 15120"

    deinterleaved_array: BitArray = np.zeros_like(arr).astype(dtype)

    # pi_j is the new index for the interleaved array
    for j in range(deinterleaved_array.shape[0]):
        pi_j = (14891 * j + 210 * j**2) % 15120
        deinterleaved_array[j] = arr[pi_j]

    return deinterleaved_array


def channel_interleave(arr: BitArray, B: int, N: int) -> BitArray:
    """Use N slots of linear shift registers to interleave the PPM symbols.

    - Input:
        - `arr`: input array / sequence
        - `B`: Base length of the linear shift registers. Such that the i-th shift register has length i*B
        - `N`: Number of rows
    """
    output: list[int] = []
    remap_indeces: list[int] = []

    # First determine the order of elements for the interleaved array .
    remap_indeces.append(0)
    for i in range(1, arr.shape[0] + B * N * (N - 1)):
        if i % N == 0:
            remap_indeces.append(i)
        else:
            remap_indeces.append(remap_indeces[i - 1] - N * B + 1)

    # Indeces < 0 indicate initial interleaver state bits, which is set at 0.
    # Indeces > the input array indicate terminal interleaver state bits, which are also set to 0

    # When the final bit of the input sequence is inserted into the interleaver,
    # The interleaver needs to be ran another B*N*(N-1) times to finalize the interleaving.
    for i in remap_indeces:
        if i < 0 or i >= arr.shape[0]:
            output.append(0)
        else:
            output.append(arr[i])

    output_array: BitArray = np.array(output, dtype=int)

    return output_array


def get_remap_indices(input_array: BitArray, B: int, N: int):
    remap_indeces: list[int] = []

    remap_indeces.append(0)
    for i in range(1, input_array.shape[0] + B * N * (N - 1)):
        if i % N == 0:
            remap_indeces.append(i)
        else:
            remap_indeces.append(remap_indeces[i - 1] + N * B + 1)

    return remap_indeces


def channel_deinterleave(arr: BitArray, B: int, N: int) -> BitArray:
    """Use N slots of linear shift registers to interleave the PPM symbols.

    - Input:
        - `arr`: input array / sequence
        - `B`: Base length of the linear shift registers. Such that the i-th shift register has length i*B
        - `N`: Number of rows
    """
    arr = np.array(arr, dtype=int)
    output: list[npt.NDArray[np.int_]] = []
    interleaver_remap_indices = get_remap_indices(arr, B, N)

    # Indeces < 0 indicate initial interleaver state bits, which is set at 0.
    # Indeces > the input array indicate terminal interleaver state bits, which are also set to 0

    # When the final bit of the input sequence is inserted into the interleaver,
    # The interleaver needs to be ran another B*N*(N-1) times to finalize the interleaving.
    for i in interleaver_remap_indices:
        if i < 0 or i >= arr.shape[0]:
            if len(arr.shape) == 2:
                output.append(np.zeros(arr.shape[1]))
            else:
                output.append(0)
        else:
            output.append(arr[i])

    output_arr: BitArray = np.array(output, dtype=int)

    return output_arr


def get_csm(M: int = 16) -> BitArray:
    validate_PPM_order(M)

    match M:
        case 4:
            w = np.array([0, 3, 1, 2, 1, 3, 2, 0, 0, 3, 2, 1, 0, 2, 1, 3, 1, 0, 3, 2, 3, 2, 1, 0])
        case 8:
            w = np.array([0, 3, 1, 2, 5, 4, 7, 6, 6, 7, 4, 5, 2, 1, 3, 0])
        case _:
            w = np.array([0, 2, 7, 14, 1, 2, 15, 5, 8, 4, 10, 2, 14, 3, 14, 11])

    return w


def slot_map(ppm_symbols: npt.NDArray[np.int_] | list[int], M: int, insert_guardslots: bool = True) -> BitArray:
    """Convert each PPM symbol to a list of ones and zeros, where the one indicates the position of the PPM pulse.

    For example, with a PPM order of 4, and a PPM symbol 3, the slot mapped vector would be [0, 0, 0, 1, 0]"""
    # Input validation
    validate_PPM_order(M)

    if not isinstance(ppm_symbols, np.ndarray):
        ppm_symbols = np.array(ppm_symbols, dtype=int)

    # Check if PPM symbols are consistent with the PPM order
    if not np.all(ppm_symbols < M):
        raise ValueError(f"All PPM symbols should be smaller than {M}")

    slot_mapped: BitArray = np.zeros((len(ppm_symbols), M), dtype=int)
    for j in range(len(ppm_symbols)):
        slot_mapped[j, ppm_symbols[j]] = 1

    # Insert guard slot (M / 4 zeros) for each slot map, if applicable
    if insert_guardslots:
        slot_mapped = np.hstack((slot_mapped, np.zeros((M // 4, slot_mapped.shape[0]), dtype=int).T))

    return slot_mapped
