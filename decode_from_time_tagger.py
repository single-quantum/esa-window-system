import pickle
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import TimeTagger

from core.BCJR_decoder_functions import ppm_symbols_to_bit_array
from core.data_converter import payload_to_bit_sequence
from core.demodulation_functions import demodulate
from core.encoder_functions import map_PPM_symbols
from core.scppm_decoder import decode
from core.utils import flatten
from ppm_parameters import (CORRELATION_THRESHOLD, DEBUG_MODE, MESSAGE_IDX,
                            USE_INNER_ENCODER, USE_RANDOMIZER)

"""Read time tagger files from the Swabian Time Tagger Ultra. Required software for the time tagger can be found here:
https://www.swabianinstruments.com/time-tagger/downloads/ . """


def get_time_events_from_tt_file(time_events_filename: str, num_channels: int, get_time_events_per_channel=True, **kwargs):
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
    time_stamps_per_channel = [[] for _ in range(num_channels)]

    while not buffer_empty:
        data = fr.getData(1000)
        events = data.getTimestamps()
        if events.size == 0:
            buffer_empty = True
            break

        channels = data.getChannels()

        events_per_channel = [list(filter(lambda e: e[1] == i, zip(events, channels)))
                              for i in range(1, num_channels+1)]

        time_stamps.append(events)
        for i in range(num_channels):
            time_stamps_per_channel[i].append(list(map(lambda e: e[0], events_per_channel[i])))

    num_items = 0
    for i in range(num_channels):
        num_items += len(flatten(time_stamps_per_channel[i]))

    time_stamps = flatten(time_stamps)
    # Time stamps from the time tagger are in picoseconds, but the rest of the code uses seconds as the base unit
    time_stamps = np.array(time_stamps)
    # time_events_per_channel = np.zeros()
    # time_events_per_channel = np.array(time_stamps_per_channel)
    time_events = time_stamps * 1E-12

    return time_events, time_stamps_per_channel


use_latest_tt_file: bool = True
time_tagger_files_dir: str = 'time tagger files/'
# reference_file_path = f'jupiter_greyscale_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence'

# You can choose to manually put in the time tagger filename below, or use the last added file to the directory.
if not use_latest_tt_file:
    time_tagger_filename = time_tagger_files_dir + \
        'jupiter_tiny_greyscale_64_samples_per_slot_CSM_0_interleaved_16-37-19.ttbin'
    metadata_filepath = Path('')
else:
    time_tagger_files_path: Path = Path(__file__).parent.absolute() / time_tagger_files_dir
    tt_files = time_tagger_files_path.rglob('*.ttbin')
    files: list[Path] = [x for x in tt_files if x.is_file()]
    files = sorted(files, key=lambda x: x.lstat().st_mtime)
    time_tagger_filename = time_tagger_files_dir + re.split(r'\.\d{1}', files[-1].stem)[0] + '.2.ttbin'
    time_tagger_file_epoch = time_tagger_filename.split('_')[-1].rstrip('.2.ttbin')

    # Get metadata files
    files_list = time_tagger_files_path.rglob('*')
    metadata_files = [f for f in files_list if not f.suffix]
    metadata_filepath = list(filter(lambda f: time_tagger_file_epoch in f.name, metadata_files))[0]

if not metadata_filepath.exists():
    raise FileNotFoundError('Metadata file not found. Check path / filename.')

with open(metadata_filepath, 'rb') as f:
    metadata = pickle.load(f)
    M = metadata.get('M')
    num_samples_per_slot = metadata.get('num_samples_per_slot')
    CODE_RATE = metadata.get('CODE_RATE')
    GREYSCALE = metadata.get('GREYSCALE')
    slot_length = metadata.get('slot_length')
    symbol_length = metadata.get('symbol_length')
    PAYLOAD_TYPE = metadata.get('PAYLOAD_TYPE')
    IMG_FILE_PATH = metadata.get('IMG_FILE_PATH')
    IMG_SHAPE = metadata.get('IMG_SHAPE')
    num_slots_per_symbol = int(5 / 4 * M)


time_events, time_events_per_channel = get_time_events_from_tt_file(time_tagger_filename, 4)
# Remove duplicate timing events
time_events = np.unique(time_events)


time_events_samples = (time_events - time_events[0]) * (8.82091E9 / num_samples_per_slot) + 0.5
# print(time_events_samples[0:110])

# plt.plot(time_events_samples[0:100]%1)

# plt.plot(time_events_samples[0:100]%1.001)

# plt.plot(time_events_samples[0:100]%0.999)

# plt.show()

vals = []
dat = time_events_samples[10000:30000]
factors = np.linspace(0.99998, 1.00002, 10000)

for factor_here in factors:
    dat2 = dat % factor_here
    vals.append(np.sum(np.abs(dat2 - np.average(dat2))**2))


# plt.figure()
# plt.plot(vals)
# plt.show()

index = np.argmin(vals)
correction = factors[index]
vals = []

print(correction)

plt.figure()
plt.plot(time_events_samples[:10000] % 1 - 0.5, label='original')
plt.xlabel('#symbol')
plt.ylabel('deviation')
plt.plot(time_events_samples[:10000] % correction - 0.5, label='adjusted')
plt.legend()
plt.show()

slot_length *= correction
symbol_length *= correction


print(f'Number of events: {len(time_events)}')

slot_mapped_message, events_per_slot = demodulate(time_events[:550000], M, slot_length, symbol_length,
                                                  num_slots_per_symbol, debug_mode=DEBUG_MODE,
                                                  csm_correlation_threshold=CORRELATION_THRESHOLD, message_idx=MESSAGE_IDX, **{'num_samples_per_slot': num_samples_per_slot})

m = int(np.log2(M))
received_ppm_symbols = np.nonzero(slot_mapped_message)[1]
received_bits = ppm_symbols_to_bit_array(received_ppm_symbols, m)

with open('received_bit_sequence', 'wb') as f:
    pickle.dump(received_bits, f)

with open('sent_bit_sequence', 'rb') as f:
    sent_bits = pickle.load(f)

with open('sent_bit_sequence_no_csm', 'rb') as f:
    sent_bits_no_csm = pickle.load(f)

BER_before_decoding = np.sum([abs(x - y) for x, y in zip(received_bits, sent_bits)]) / len(sent_bits)
print('BER before decoding', BER_before_decoding)

information_blocks, BER_before_decoding = decode(
    slot_mapped_message, M, CODE_RATE,
    use_inner_encoder=USE_INNER_ENCODER,
    **{
        'use_cached_trellis': False,
        'num_events_per_slot': events_per_slot,
        'use_randomizer': USE_RANDOMIZER,
        'sent_bit_sequence_no_csm': sent_bits_no_csm,
        'sent_bit_sequence': sent_bits
    })

sent_img_array: npt.NDArray[np.int_] = np.array([])
img_arr: npt.NDArray[np.int_] = np.array([])
CMAP = ''

if PAYLOAD_TYPE == 'image':
    # compare to original image
    sent_message = payload_to_bit_sequence(PAYLOAD_TYPE, filepath=IMG_FILE_PATH)

    if GREYSCALE:
        sent_img_array = map_PPM_symbols(sent_message, 8)
        pixel_values = map_PPM_symbols(information_blocks, 8)
        img_arr = pixel_values[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
        CMAP = 'Greys'
    else:
        img_arr = information_blocks.flatten()[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
        CMAP = 'binary'


# In the case of a greyscale image, each pixel has a value from 0 to 255.
# This would be the same as saying that each pixel is a symbol, which should be mapped to an 8 bit sequence.
# if GREYSCALE:
#     sent_message = ppm_symbols_to_bit_array(sent_img_array.flatten(), 8)
# else:
#     sent_message = sent_img_array.flatten()

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
