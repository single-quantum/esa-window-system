import numpy as np
import numpy.typing as npt
from scipy.ndimage import shift

from shift_register import CRC


def get_asm_bit_arr(asm_hex: str = '1ACFFC1D') -> npt.NDArray:
    """Returns the binary representation of the Attached Synchronisation Markerk, as defined by the CCSDS. """
    ASM_binary = np.binary_repr(int(asm_hex, base=16), width=32)
    ASM_arr = np.array([int(i) for i in ASM_binary])

    return ASM_arr.astype(int)


def prepend_asm(arr) -> npt.NDArray:
    """Prepend `arr` with the Attached Synchronization Marker. """
    ASM_arr = get_asm_bit_arr()
    # Tile the ASM, such that it vertically, so that it can be stacked to the reshaped array
    asm_bit_array = np.tile(ASM_arr, arr.shape[0]).reshape(arr.shape[0], -1)

    return np.hstack((asm_bit_array, arr))


def generate_pseudo_randomized_sequence(seed: list = [1] * 8):
    """Generate the pseudo randomized sequence, as defined in the CCSDS standard. """
    sequence = []

    sequence.append(seed[-1])

    addition_1 = seed[4] ^ seed[7]
    addition_2 = seed[2] ^ addition_1
    addition_3 = seed[0] ^ addition_2

    output = shift(seed, 1).tolist()
    output[0] = addition_3

    sequence.append(output[-1])
    i = 2

    while i < 256:
        # print('input', output)
        addition_1 = output[4] ^ output[7]
        addition_2 = output[2] ^ addition_1
        addition_3 = output[0] ^ addition_2

        output = shift(output, 1).tolist()
        output[0] = addition_3

        sequence.append(output[-1])

        i += 1

    return sequence


def append_CRC(arr: npt.NDArray):
    # Fill the input array `arr` with 32 zeros, so that the CRC can be attached
    CRC_size = 32
    arr = np.pad(arr, (0, CRC_size))

    seed = [1] * CRC_size

    # Initialize the CRC shift register
    sr = CRC(seed, [3, 14, 18, 29])
    for i in range(1, CRC_size + 1):
        sr.next(arr[-i] ^ sr.state[-1])

    # Attach to the arr
    arr[-CRC_size:] = sr.state

    # Add the two termination bits
    return np.pad(arr, (0, 2))


def slicer(arr: npt.NDArray, include_crc=False) -> npt.NDArray:
    if include_crc:
        information_block_size = 5006
    else:
        information_block_size = 5038

    if arr.shape[0] % information_block_size != 0:
        # Number of bits that need to be added such that the array is a multiple of the information block size
        num_pad_bits = information_block_size - arr.shape[0] % information_block_size
        arr = np.hstack((arr, np.zeros(num_pad_bits, dtype=int)))

    arr = arr.reshape((-1, information_block_size))
    return arr


def zero_terminate(arr: npt.NDArray) -> npt.NDArray:
    zero_bits = np.zeros((arr.shape[0], 2), dtype=int)
    return np.hstack((arr, zero_bits))


def accumulate(arr: npt.NDArray) -> npt.NDArray:
    """Accumulate XOR-wise `arr` with itself. """
    n_j = np.zeros_like(arr)
    n_j[0] = arr[0]
    for j in range(1, n_j.shape[0]):
        n_j[j] = n_j[j - 1] ^ arr[j]

    return n_j


def convolve(arr, initial_state=[0, 0]):
    """Use a convolutional shift register to generate a convoluted codeword. """
    # Number of sliding windows that are iterated over to generate the
    # convolutional codeword
    num_windows: int = len(arr)
    convolutional_codeword = np.zeros(3 * num_windows, dtype=int)

    # For now, I pad the SCCM encoder input block with 0 zeros, but these
    # should come from the next SCPPM block in the sequence.
    # arr = np.pad(arr, (2, 0))
    arr = np.hstack((tuple(reversed(initial_state)), arr))
    # arr = np.pad(arr, (0, 2))

    for i in range(num_windows):
        f = arr[i:i + 3]

        h = [f[2] ^ f[0], f[0] ^ f[1] ^ f[2], f[0] ^ f[1] ^ f[2]]
        convolutional_codeword[3 * i:3 * i + 3] = h

    terminal_state = tuple(np.flip(f[1:]))

    return convolutional_codeword, terminal_state


def map_PPM_symbols(arr, S: int, m: int):
    output_arr = np.zeros(S, dtype=int)
    for s in range(S):
        q = np.zeros(m)
        for a in range(m):
            q[a] = 2**(m - a - 1) * arr[m * s + a]
        output_arr[s] = np.sum(q)

    return output_arr


def bit_interleave(arr: npt.ArrayLike) -> npt.NDArray[np.int_]:
    """Shuffle some bits around to make a so-called bit-interleaved codeword.

    Note: works only with 15120 element arrays. The modulo 15120 is hard coded for a reason.
    This is because the permutation polynomial in `bit_deinterleave` only works properly with 15120 element arrays.
    For any other length array, the interleaved array might not be invertable. """

    assert len(arr) == 15120, "Input array should have length 15120"

    interleaved_output = np.zeros_like(arr).astype(int)

    # pi_j is the new index for the interleaved array
    for j in range(interleaved_output.shape[0]):
        pi_j = (11 * j + 210 * j**2) % 15120
        interleaved_output[j] = arr[pi_j]

    return interleaved_output


def bit_deinterleave(arr: npt.ArrayLike) -> npt.NDArray[np.int_]:
    """De-interleave the interleaved array `arr`.

    Note: works only with 15120 element arrays. """

    assert len(arr) == 15120, "Input array should have length 15120"

    deinterleaved_array = np.zeros_like(arr).astype(int)

    # pi_j is the new index for the interleaved array
    for j in range(deinterleaved_array.shape[0]):
        pi_j = (14891 * j + 210 * j**2) % 15120
        deinterleaved_array[j] = arr[pi_j]

    return deinterleaved_array


def channel_interleave(arr, B, N):
    """Use N slots of linear shift registers to interleave the PPM symbols.

    - Input:
        - `arr`: input array / sequence
        - `B`: Base length of the linear shift registers. Such that the i-th shift register has length i*B
        - `N`: Number of rows
    """
    output = []
    remap_indeces = []

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

    return np.array(output, dtype=int)


def channel_deinterleave(arr: npt.ArrayLike, B: int, N: int) -> list[int]:
    """Use N slots of linear shift registers to interleave the PPM symbols.

    - Input:
        - `arr`: input array / sequence
        - `B`: Base length of the linear shift registers. Such that the i-th shift register has length i*B
        - `N`: Number of rows
    """
    arr = np.array(arr, dtype=int)
    output = []
    remap_indeces = []

    remap_indeces.append(0)
    for i in range(1, arr.shape[0] + B * N * (N - 1)):
        if i % N == 0:
            remap_indeces.append(i)
        else:
            remap_indeces.append(remap_indeces[i - 1] + N * B + 1)

    # Indeces < 0 indicate initial interleaver state bits, which is set at 0.
    # Indeces > the input array indicate terminal interleaver state bits, which are also set to 0

    # When the final bit of the input sequence is inserted into the interleaver,
    # The interleaver needs to be ran another B*N*(N-1) times to finalize the interleaving.
    for i in remap_indeces:
        if i < 0 or i >= arr.shape[0]:
            output.append(0)
        else:
            output.append(arr[i])

    return output

def get_csm(M=16):
    match M:
        case 4:
            w = np.array([0, 3, 1, 2, 1, 3, 2, 0, 0, 3, 2, 1, 0, 2, 1, 3, 1, 0, 3, 2, 3, 2, 1, 0])
        case 8:
            w = np.array([0, 3, 1, 2, 5, 4, 7, 6, 6, 7, 4, 5, 2, 1, 3, 0])
        case _:
            w = np.array([0, 2, 7, 14, 1, 2, 15, 5, 8, 4, 10, 2, 14, 3, 14, 11])

    return w