import numpy as np
import cython
from cython.parallel import prange
import numpy.typing as npt


def get_num_events_per_slot(
        csm_times: npt.NDArray[np.float_],
        peak_locations: npt.NDArray[np.float_],
        CSM: npt.NDArray[np.int_],
        symbols_per_codeword: cython.int,
        slot_length: cython.double,
        M: cython.int) -> npt.NDArray[np.int_]:
    """This function determines how many detection events there were for each slot. """

    # The factor 5/4 is determined by the protocol, which states that there
    # shall be M/4 guard slots for each PPM symbol.
    num_slots_per_codeword: cython.int = int((symbols_per_codeword + len(CSM)) * 5 / 4 * M)
    num_rows: cython.Py_ssize_t = csm_times.shape[0]

    my_type = cython.fused_type(cython.int, cython.double, cython.longlong)
    num_events_per_slot: npt.NDArray[np.int_] = np.zeros((num_rows, num_slots_per_codeword), dtype=np.int_)

    for i in range(len(csm_times)):
        csm_time: cython.double = csm_times[i]

        # Preselect those detection peaks that are within the csm times
        if i < num_rows - 1:
            idx_arr_1: npt.NDArray[bool] = peak_locations >= csm_times[i]
            idx_arr_2: npt.NDArray[bool] = peak_locations < csm_times[i + 1]
            message_peak_locations = peak_locations[
                (idx_arr_1) & (idx_arr_2)
            ]
        else:
            idx_arr: npt.NDArray[bool] = peak_locations >= csm_times[i]
            message_peak_locations = peak_locations[idx_arr]

        slot_starts: cython.double[:] = csm_time + np.arange(num_slots_per_codeword+1)*slot_length

        num_events: cython.int
        for j in range(num_slots_per_codeword):
            idx_arr_1: cython.Py_ssize_t = message_peak_locations >= slot_starts[j]
            idx_arr_2: cython.Py_ssize_t = message_peak_locations < slot_starts[j+1]
            events: cython.double[:] = message_peak_locations[
                (idx_arr_1) & (idx_arr_2)
            ]

            num_events = events.shape[0]
            num_events_per_slot[i, j] = num_events

    num_events_per_slot_return: cython.int[:] = num_events_per_slot.flatten().astype(int)
    return num_events_per_slot_return
