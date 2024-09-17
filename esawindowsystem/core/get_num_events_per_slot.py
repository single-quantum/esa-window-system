import numpy as np
import numpy.typing as npt
from numba import jit, njit, prange, types
from numba.pycc import CC
from math import ceil

cc = CC('numba_utils')
# Uncomment the following line to print out the compilation steps
cc.verbose = True


@cc.export('get_num_events', 'i4[:, :](i4, i4[:, :], i4, f8[:], f8[:])')
@jit(parallel=True, fastmath=True)
def get_num_events(
        i: int,
        num_events_per_slot: npt.NDArray[np.int_],
        num_slots_per_codeword: int,
        message_peak_locations: npt.NDArray[np.float64],
        slot_starts: npt.NDArray[np.float64]) -> npt.NDArray[np.int_]:

    for j in prange(num_slots_per_codeword):
        idx_arr_1: npt.NDArray[np.bool] = message_peak_locations >= slot_starts[j]
        idx_arr_2: npt.NDArray[np.bool] = message_peak_locations < slot_starts[j + 1]
        events = message_peak_locations[
            (idx_arr_1) & (idx_arr_2)
        ]

        num_events: int = events.shape[0]
        num_events_per_slot[i, j] = num_events

    return num_events_per_slot


@cc.export('get_num_events_numba', 'i4[:](f8[:], f8[:], i4)')
@njit()
def get_num_events_2(
        message_peak_locations: npt.NDArray[np.float64],
        slot_starts: npt.NDArray[np.float64],
        chunk_size: int = 1000) -> npt.NDArray[np.int32]:

    num_slots: int = slot_starts.shape[0]
    chunk_size = 5000
    num_chunks: int = ceil(num_slots/chunk_size)

    num_events = np.empty((num_slots,), dtype=np.int32)

    message_peak_locations_copied = np.empty((chunk_size, message_peak_locations.shape[0]))
    message_peak_locations_copied[:] = message_peak_locations
    message_peak_locations_copied = np.transpose(message_peak_locations_copied)

    i = 0
    for i in prange(num_chunks):
        remainder = slot_starts.shape[0] - i*chunk_size

        if remainder < chunk_size:
            A = message_peak_locations_copied[:, :remainder] >= slot_starts[i*chunk_size:(i+1)*chunk_size]
            B = message_peak_locations_copied[:, :remainder] < slot_starts[i*chunk_size:(i+1)*chunk_size]
        else:
            A = message_peak_locations_copied >= slot_starts[i*chunk_size:(i+1)*chunk_size]
            B = message_peak_locations_copied < slot_starts[i*chunk_size:(i+1)*chunk_size]

        B = np.roll(B, -1)

        num_events[i*chunk_size:(i+1)*chunk_size] = np.sum((A & B), axis=0)

    return num_events


if __name__ == "__main__":
    cc.compile()
