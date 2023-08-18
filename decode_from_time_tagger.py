import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import TimeTagger

from core.BCJR_decoder_functions import ppm_symbols_to_bit_array
from core.demodulation_functions import demodulate
from core.encoder_functions import map_PPM_symbols
from ppm_parameters import (CODE_RATE, GREYSCALE, IMG_SHAPE, PAYLOAD_TYPE, IMG_FILE_PATH,
                            num_slots_per_symbol, slot_length, symbol_length, M, num_samples_per_slot,
                            DEBUG_MODE, CORRELATION_THRESHOLD, MESSAGE_IDX)
import numpy.typing as npt

from core.scppm_decoder import decode
from core.utils import flatten
from core.data_converter import payload_to_bit_sequence

"""Read time tagger files from the Swabian Time Tagger Ultra. Required software for the time tagger can be found here:
https://www.swabianinstruments.com/time-tagger/downloads/ . """


def get_time_events_from_tt_file(time_events_filename: str, **kwargs):
    """Open the `time_events_filename` with the TimeTagger.FileReader class and retrieve events.

    Can either read out the entire buffer or read out a given number of events. """
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
    # Time stamps from the time tagger are in picoseconds, but the rest of the code uses seconds as the base unit
    time_stamps = np.array(time_stamps)
    time_events = time_stamps * 1E-12

    return time_events


use_latest_tt_file: bool = False
time_tagger_files_dir: str = 'time tagger files/'
reference_file_path = f'jupiter_greyscale_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence'

# You can choose to manually put in the time tagger filename below, or use the last added file to the directory.
if not use_latest_tt_file:
    time_tagger_filename = time_tagger_files_dir + \
        'jupiter_tiny_greyscale_64_samples_per_slot_CSM_0_interleaved_16-37-19.ttbin'
else:
    time_tagger_files_path: Path = Path(__file__).parent.absolute() / time_tagger_files_dir
    tt_files = time_tagger_files_path.rglob('*.ttbin')
    files: list[Path] = [x for x in tt_files if x.is_file()]
    files = sorted(files, key=lambda x: x.lstat().st_mtime)
    time_tagger_filename = time_tagger_files_dir + re.split(r'\.\d{1}', files[-1].stem)[0] + '.ttbin'


time_events = get_time_events_from_tt_file(time_tagger_filename)
# Remove duplicate timing events
time_events = np.unique(time_events)

print(f'Number of events: {len(time_events)}')

slot_mapped_message = demodulate(time_events[:100000], M, slot_length, symbol_length,
                                 num_slots_per_symbol, debug_mode=DEBUG_MODE,
                                 csm_correlation_threshold=CORRELATION_THRESHOLD, message_idx=MESSAGE_IDX)


information_blocks, BER_before_decoding = decode(
    slot_mapped_message, M, CODE_RATE,
    **{'use_cached_trellis': False, })

sent_img_array: npt.NDArray[np.int_] = np.array([])
img_arr: npt.NDArray[np.int_] = np.array([])
CMAP = ''

if PAYLOAD_TYPE == 'image':
    # compare to original image
    sent_img_array = payload_to_bit_sequence(PAYLOAD_TYPE, filepath=IMG_FILE_PATH)

    if GREYSCALE:
        pixel_values = map_PPM_symbols(information_blocks, 8)
        img_arr = pixel_values[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
        CMAP = 'Greys'
    else:
        img_arr = information_blocks.flatten()[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
        CMAP = 'binary'


# In the case of a greyscale image, each pixel has a value from 0 to 255.
# This would be the same as saying that each pixel is a symbol, which should be mapped to an 8 bit sequence.
if GREYSCALE:
    sent_message = ppm_symbols_to_bit_array(sent_img_array.flatten(), 8)
else:
    sent_message = sent_img_array.flatten()

if len(information_blocks) < len(sent_message):
    BER_after_decoding = np.sum(np.abs(information_blocks -
                                sent_message[:len(information_blocks)])) / len(information_blocks)
else:
    BER_after_decoding = np.sum(
        np.abs(information_blocks[:len(sent_message)] - sent_message)) / len(sent_message)

print(f'BER after decoding: {BER_after_decoding }. ')

if PAYLOAD_TYPE == 'image':
    fig, axs = plt.subplots(1, 2)
    plt.suptitle('Sent / decoded payload comparison')
    axs[0].imshow(sent_img_array.reshape(IMG_SHAPE[0], IMG_SHAPE[1]), cmap=CMAP)
    axs[0].set_title('Sent image')
    axs[0].set_xlabel('Pixel number (x)')
    axs[0].set_ylabel('Pixel number (y)')

    axs[1].imshow(img_arr, cmap=CMAP)
    axs[1].set_xlabel('Pixel number (x)')
    axs[1].set_title('Decoded image using SQ cam')
    plt.show()

print('Done')
