"""This script is used to acquire time tags from the Swabian time tagger ultra. """
from datetime import datetime
from time import sleep

import TimeTagger

from ppm_parameters import CODE_RATE, M, num_samples_per_slot

num_channels = 4
# channels = [i+1 for i in range(num_channels)]
channels = [1, 2, 3, 4]

tagger = TimeTagger.createTimeTagger(resolution=TimeTagger.Resolution.Standard)
serial = tagger.getSerial()
print(f'Connected to time tagger {serial}')

for i in channels:
    tagger.setDeadtime(i, 10000)
    tagger.setTriggerLevel(i, 0.10)
tagger.sync()

sleep(2)
db_att = 0

current_time = datetime.now()
print('Current time: ', current_time)
formatted_time = current_time.strftime("%H-%M-%S")
window_size_secs = 20E-3
window_size_ps = window_size_secs * 1E12  # Window time in ps

cr = str(CODE_RATE).replace('/', '-')

filewriter = TimeTagger.FileWriter(
    tagger,
    f'time tagger files/jupiter_tiny_greyscale_{num_samples_per_slot}-sps_{M}-PPM_{cr}-code-rate_{formatted_time}',
    channels=channels)
filewriter.startFor(int(window_size_ps), clear=True)
filewriter.waitUntilFinished()

num_events = filewriter.getTotalEvents()

print(f'{num_events} events written to disk. ')
print(f'Events per second: {num_events*1/window_size_secs:.3e}')
