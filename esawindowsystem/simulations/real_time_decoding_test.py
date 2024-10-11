import pickle
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
# import TimeTagger
from numpy.random import default_rng
from PIL import Image
from scipy.signal import find_peaks

from esawindowsystem.core.BCJR_decoder_functions import \
    ppm_symbols_to_bit_array
from esawindowsystem.core.demodulation_functions import demodulate
from esawindowsystem.core.encoder_functions import map_PPM_symbols
from esawindowsystem.core.scppm_decoder import DecoderError, decode
from esawindowsystem.core.trellis import Trellis
from esawindowsystem.core.utils import flatten
from esawindowsystem.generate_awg_pattern import generate_awg_pattern
from esawindowsystem.ppm_parameters import (CODE_RATE, GREYSCALE,
                                            IMG_FILE_PATH, IMG_SHAPE, M,
                                            num_samples_per_slot,
                                            num_slots_per_symbol,
                                            sample_size_awg, slot_length,
                                            symbol_length)


import TimeTagger
import numpy.typing as npt
from time import time
from time import sleep


def flatten(list_of_lists: list[list] | list[npt.NDArray]) -> list:
    """Convert a list of lists to a flat (1D) list. """
    return [i for sublist in list_of_lists for i in sublist]


# Create a TimeTagger instance to control your hardware
tagger = TimeTagger.createTimeTaggerVirtual()

# Enable the test signal on channels 1 and 2
tagger.setTestSignal([1, 2], True)

event_buffer_size = 1000000

stream = TimeTagger.TimeTagStream(tagger=tagger,
                                  n_max_events=event_buffer_size,
                                  channels=[1, 2])

format_string = '{:>8} | {:>17} | {:>7} | {:>14} | {:>13}'
print(format_string.format('TAG #', 'EVENT TYPE',
      'CHANNEL', 'TIMESTAMP (ps)', 'MISSED EVENTS'))
print('---------+-------------------+---------+----------------+--------------')
event_name = ['0 (TimeTag)', '1 (Error)', '2 (OverflowBegin)',
              '3 (OverflowEnd)', '4 (MissedEvents)']

stream.startFor(int(1E12))
event_counter = 0
chunk_counter = 1


# Generate virtual counts
# tagger.setReplaySpeed(0.15)
tagger.replay(
    'herbig-haro-211_10-sps_16-PPM_2-3-code-rate_11-56-00_1701428160.ttbin')
# tagger.replay(
#     'herbig-haro-211_10-sps_16-PPM_2-3-code-rate_11-56-00_1701428160.ttbin')

timestamps_array = []
running_time = 0

while stream.isRunning():
    start_time = time()
    # getData() does not return timestamps, but an instance of TimeTagStreamBuffer
    # that contains more information than just the timestamp
    data = stream.getData()

    if data.size == event_buffer_size:
        print('TimeTagStream buffer is filled completely. Events arriving after the buffer has been filled have been discarded. Please increase the buffer size not to miss any events.')

    if data.size > 0:
        # With the following methods, we can retrieve a numpy array for the particular information:
        channel = data.getChannels()            # The channel numbers
        timestamps = data.getTimestamps()       # The timestamps in ps
        # TimeTag = 0, Error = 1, OverflowBegin = 2, OverflowEnd = 3, MissedEvents = 4
        overflow_types = data.getEventTypes()
        # The numbers of missed events in case of overflow
        missed_events = data.getMissedEvents()

        print(format_string.format(*" "*5))
        heading = ' Start of data chunk {} with {} events '.format(
            chunk_counter, data.size)
        extra_width = 69 - len(heading)
        print('{} {} {}'.format("="*(extra_width//2),
              heading, "="*(extra_width - extra_width//2)))
        print(format_string.format(*" "*5))

        print(format_string.format(event_counter + 1,
              event_name[overflow_types[0]], channel[0], timestamps[0], missed_events[0]))
        if data.size > 1:
            print(format_string.format(event_counter + 2,
                  event_name[overflow_types[1]], channel[1], timestamps[1], missed_events[1]))
        if data.size > 3:
            print(format_string.format(*["..."]*5))
        if data.size > 2:
            print(format_string.format(event_counter + data.size,
                  event_name[overflow_types[-1]], channel[-1], timestamps[-1]*1E-12, missed_events[-1]))

        timestamps_array.append(timestamps)

        event_counter += data.size
        chunk_counter += 1

    sleep(0.1)
    end_time = time()

    running_time += (end_time - start_time)
    print(f"start time: \t {start_time} \t end time: \t " +
          f"{end_time} \t running time: \t {running_time}")

    if running_time > 2:
        break


timestamps_array = flatten(timestamps_array)


TimeTagger.freeTimeTagger(tagger)


slot_mapped_message, _ = demodulate(
    peak_locations[:200000], M, slot_length, symbol_length, csm_correlation_threshold=0.75, **{'debug_mode': False})


information_blocks: npt.NDArray[np.int_] = np.array([])


information_blocks, BER_before_decoding, _ = decode(
    slot_mapped_message, M, CODE_RATE, CHANNEL_INTERLEAVE=True, BIT_INTERLEAVE=True, use_inner_encoder=True,
    **{
        'use_cached_trellis': False,
        # 'cached_trellis_file_path': cached_trellis_file_path,
        'user_settings': {'reference_file_path': reference_file_path},
    })
