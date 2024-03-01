"""This script is used to acquire time tags from the Swabian time tagger ultra. """
from datetime import datetime
from time import sleep
from pathlib import Path
import pickle

import TimeTagger

from ppm_parameters import CODE_RATE, M, num_samples_per_slot, IMG_FILE_PATH, GREYSCALE, slot_length, symbol_length, PAYLOAD_TYPE, IMG_SHAPE, USE_INNER_ENCODER, USE_RANDOMIZER

# num_channels = 4
# channels = [i+1 for i in range(num_channels)]
channels = [1, 2, 3, 4]

tagger = TimeTagger.createTimeTagger(resolution=TimeTagger.Resolution.Standard)
# sampling_time_ps = tagger.getPsPerClock()
# print(f'{1/(sampling_time_ps*1E-12):.3e}')
# print('used clock', tagger.xtra_getClockSource())
serial = tagger.getSerial()
print(f'Connected to time tagger {serial}')

for i in channels:
    tagger.setDeadtime(i, 100)
    tagger.setTriggerLevel(i, 0.10)
tagger.sync()

sleep(1)

current_time = datetime.now()
timestamp_epoch = int(datetime.timestamp(datetime.now()))
print('Current time: ', current_time)
formatted_time = current_time.strftime("%H-%M-%S")
window_size_secs = 2*3000E-6
window_size_ps = window_size_secs * 1E12  # Window time in ps

cr = str(CODE_RATE).replace('/', '-')

img_path = Path(IMG_FILE_PATH)
img_name = img_path.name.rstrip(img_path.suffix)

filewriter = TimeTagger.FileWriter(
    tagger,
    f'time tagger files/{img_name}_{num_samples_per_slot}-sps_{M}-PPM_{cr}-code-rate_{formatted_time}_{timestamp_epoch}',
    channels=channels)
filewriter.startFor(int(window_size_ps), clear=True)
filewriter.waitUntilFinished()

num_events = filewriter.getTotalEvents()
events_per_second = num_events/window_size_secs

print(f'{num_events} events written to disk. ')
print(f'Events per second: {events_per_second:.3e}')

with open('sent_bit_sequence', 'rb') as f:
    sent_bits = pickle.load(f)

with open('sent_bit_sequence_no_csm', 'rb') as f:
    sent_bits_no_csm = pickle.load(f)

# Write metadata file
with open(f'time tagger files/timetags_metadata_{timestamp_epoch}', 'wb') as f:
    metadata = {
        'M': M,
        'num_samples_per_slot': num_samples_per_slot,
        'CODE_RATE': CODE_RATE,
        'GREYSCALE': GREYSCALE,
        'slot_length': slot_length,
        'symbol_length': symbol_length,
        'PAYLOAD_TYPE': PAYLOAD_TYPE,
        'IMG_SHAPE': IMG_SHAPE,
        'IMG_FILE_PATH': IMG_FILE_PATH,
        'USE_INNER_ENCODER': USE_INNER_ENCODER,
        'USE_RANDOMIZER': USE_RANDOMIZER,
        'num_events_TT': num_events,
        'acquisition_time_TT': window_size_secs,
        'recorded_channels': channels,
        'countrate_TT': events_per_second,
        'timestamp_epoch': timestamp_epoch,
        'sent_bit_sequence': sent_bits,
        'sent_bit_sequence_no_csm': sent_bits_no_csm
    }

    pickle.dump(metadata, f)

print('Metadata epoch:', timestamp_epoch)
