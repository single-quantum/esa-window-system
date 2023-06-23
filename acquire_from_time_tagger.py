"""This script is used to acquire time tags from the Swabian time tagger ultra. """
from datetime import datetime
from time import sleep

import TimeTagger
import pint

from ppm_parameters import CODE_RATE, M, num_samples_per_slot

ureg = pint.UnitRegistry()

# input parameters
window_size_ms = 50 * ureg('milliseconds')
trigger_level = 100 * ureg('millivolt')

tagger = TimeTagger.createTimeTagger(resolution=TimeTagger.Resolution.Standard)
tagger.setDeadtime(1, 2)
serial = tagger.getSerial()
print(f'Connected to time tagger {serial}')
tagger.setTriggerLevel(1, trigger_level.to('volt').magnitude)
tagger.sync()

sleep(2)
db_att = 0

current_time = datetime.now()
print('Current time: ', current_time)
formatted_time = current_time.strftime("%H-%M-%S")
window_size_ps = window_size_ms.to('picoseconds')  # Window time in ps

cr = str(CODE_RATE).replace('/', '-')

filewriter = TimeTagger.FileWriter(
    tagger,
    f'time tagger files/jupiter_tiny_greyscale_{num_samples_per_slot}-sps_{M}-PPM_{cr}-code-rate_{formatted_time}',
    channels=[1])
filewriter.startFor(int(window_size_ps.magnitude), clear=True)
filewriter.waitUntilFinished()

num_events = filewriter.getTotalEvents()

print(f'{num_events} events written to disk. ')
print(f"Events per second: {num_events/window_size_ms.to('seconds')}")
