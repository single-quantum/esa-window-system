import numpy as np

from esawindowsystem.core.demodulation_functions import get_num_events


def test_get_num_events():
    slot_starts = np.array([0, 1E-6, 2E-6])
    num_slots_per_codeword = 10
    num_events_per_slot = np.zeros((2, num_slots_per_codeword))
    message_peak_locations = np.array([0, 0.1E-7, 1.1E-6, 2.1E-6, 2.2E-6, 2.3E-6])

    result = get_num_events(0, num_events_per_slot,
                            num_slots_per_codeword, message_peak_locations, slot_starts)

    assert result[0, 0] == 2
