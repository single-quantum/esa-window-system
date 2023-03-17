import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import TimeTagger

from demodulation_functions import demodulate
from encoder_functions import map_PPM_symbols
from ppm_parameters import (BIT_INTERLEAVE, CHANNEL_INTERLEAVE, CODE_RATE,
                            GREYSCALE, IMG_SHAPE, B_interleaver, N_interleaver,
                            m)
from scppm_decoder import DecoderError, decode
from utils import flatten


def get_time_events_from_tt_file(time_events_filename: str, **kwargs):
    """Open the `time_events_filename` with the TimeTagger.FileReader class and retrieve events. """
    fr = TimeTagger.FileReader(time_events_filename)

    if num_events := kwargs.get('num_events'):
        data = fr.getData(num_events)
        time_stamps = data.getTimestamps()
        time_events = time_stamps * 1E-12

        return time_events

    buffer_empty: bool = False

    # It is a bit awkward to work with a list here, then flatten it and
    # recast it to a numpy array, but it is much faster than working with np.hstack / np.append
    time_stamps = []

    while not buffer_empty:
        data = fr.getData(1000)
        events = data.getTimestamps()
        if events.size == 0:
            buffer_empty = True
        else:
            time_stamps.append(events)

    time_stamps = flatten(time_stamps)
    time_stamps = np.array(time_stamps)

    time_events = time_stamps * 1E-12

    return time_events


use_latest_tt_file: bool = False
time_tagger_files_dir: str = 'time tagger files/'

if use_latest_tt_file:
    time_tagger_files_path: Path = Path(__file__).parent.absolute() / time_tagger_files_dir
    tt_files = time_tagger_files_path.rglob('*.ttbin')
    files: list[Path] = [x for x in tt_files if x.is_file()]
    files = sorted(files, key=lambda x: x.lstat().st_mtime)
    time_events_filename = time_tagger_files_dir + re.split(r'\.\d{1}', files[-1].stem)[0] + '.ttbin'
else:
    time_events_filename = time_tagger_files_dir + 'jupiter_tiny_greyscale_16_samples_per_slot_CSM_0_interleaved_15-56-53.ttbin'


time_events = get_time_events_from_tt_file(time_events_filename)

print(f'Number of events: {len(time_events)}')

ppm_mapped_message = demodulate(time_events[100000:300000])

information_blocks, BER_before_decoding = decode(
    ppm_mapped_message, B_interleaver, N_interleaver, m, CHANNEL_INTERLEAVE, BIT_INTERLEAVE, CODE_RATE,
    **{'use_cached_trellis': False, })


if GREYSCALE:
    pixel_values = map_PPM_symbols(information_blocks, 8)
    img_arr = pixel_values[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
    CMAP = 'Greys'
    MODE = "L"
    IMG_MODE = 'L'
else:
    img_arr = information_blocks.flatten()[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
    CMAP = 'binary'
    MODE = "1"
    IMG_MODE = '1'

plt.figure()
plt.imshow(img_arr)
plt.show()
