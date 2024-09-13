from numba import jit, prange, types
import numpy as np
import numpy.typing as npt
from numba.pycc import CC

cc = CC('numba_utils')
# Uncomment the following line to print out the compilation steps
cc.verbose = True


@cc.export('get_num_events', 'i4[:, :](i4, i4[:, :], i4, f8[:], f8[:])')
@jit(parallel=True, fastmath=True)
def get_num_events(
        i: int,
        num_events_per_slot: npt.NDArray[np.int_],
        num_slots_per_codeword: int,
        message_peak_locations: npt.NDArray[np.float_],
        slot_starts: npt.NDArray[np.float_]) -> npt.NDArray[np.int_]:

    for j in prange(num_slots_per_codeword):
        idx_arr_1: npt.NDArray[np.bool] = message_peak_locations >= slot_starts[j]
        idx_arr_2: npt.NDArray[np.bool] = message_peak_locations < slot_starts[j+1]
        events = message_peak_locations[
            (idx_arr_1) & (idx_arr_2)
        ]

        num_events: int = events.shape[0]
        num_events_per_slot[i, j] = num_events

    return num_events_per_slot


if __name__ == "__main__":
    cc.compile()
