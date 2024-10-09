import numpy as np
import pytest

from esawindowsystem.core.demodulation_functions import get_num_events


@pytest.fixture
def short_message():
    num_slots_per_codeword = 1000
    slot_length = 1E-6
    slot_starts = np.arange(0, (num_slots_per_codeword + 1) * slot_length, slot_length)

    message_peak_locations = np.array([0, 0.1E-7, 1.1E-6, 2.1E-6, 2.2E-6, 2.3E-6])

    return num_slots_per_codeword, slot_starts, message_peak_locations


def test_get_num_events(short_message):
    num_slots_per_codeword, slot_starts, message_peak_locations = short_message

    num_events_per_slot = np.zeros((1, num_slots_per_codeword), dtype=int)

    result = get_num_events(0, num_events_per_slot,
                            num_slots_per_codeword, message_peak_locations, slot_starts)

    assert np.all(result[0, :3] == np.array([2, 1, 3]))
