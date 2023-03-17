"""This script is used to acquire time tags from the Swabian time tagger ultra. """
from datetime import datetime
from time import sleep

import TimeTagger

from ppm_parameters import CODE_RATE, M, num_samples_per_slot

tagger = TimeTagger.createTimeTagger(resolution=TimeTagger.Resolution.Standard)
tagger.setDeadtime(1, 2)
serial = tagger.getSerial()
print(f'Connected to time tagger {serial}')
tagger.setTriggerLevel(1, 0.10)
tagger.sync()

sleep(2)
db_att = 0

current_time = datetime.now()
print('Current time: ', current_time)
formatted_time = current_time.strftime("%H-%M-%S")
window_size_secs = 50E-3
window_size_ps = window_size_secs * 1E12  # Window time in ps

cr = str(CODE_RATE).replace('/', '-')

filewriter = TimeTagger.FileWriter(
    tagger,
    f'time tagger files/jupiter_tiny_greyscale_{num_samples_per_slot}-sps_{M}-PPM_{cr}-code-rate_{formatted_time}',
    channels=[1])
filewriter.startFor(int(window_size_ps), clear=True)
filewriter.waitUntilFinished()

num_events = filewriter.getTotalEvents()

print(f'{num_events} events written to disk. ')
print(f'Events per second: {num_events*1/window_size_secs}')
