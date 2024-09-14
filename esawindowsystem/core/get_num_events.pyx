import cython

def get_num_events(
        i: cython.Py_ssize_t,
        num_events_per_slot: cython.int[:, :],
        num_slots_per_codeword: cython.Py_ssize_t,
        message_peak_locations: cython.double[:],
        slot_starts: cython.double[:]) -> cython.int[:, :]:

    num_events_per_slot_view: cython.int[:, :] = num_events_per_slot
    j: cython.Py_ssize_t
    z: cython.Py_ssize_t
    z_max: cython.Py_ssize_t = message_peak_locations.shape[0]

    for j in range(num_slots_per_codeword):
        for z in range(z_max):
            if message_peak_locations[z] >= slot_starts[j]:
                continue
        idx_arr_1: npt.NDArray[cython.bint] = message_peak_locations >= slot_starts[j]
        idx_arr_2: npt.NDArray[cython.bint] = message_peak_locations < slot_starts[j+1]
        events = message_peak_locations[(idx_arr_1) & (idx_arr_2)]

        num_events: cython.int = events.shape[0]
        num_events_per_slot_view[i, j] = num_events

    return num_events_per_slot