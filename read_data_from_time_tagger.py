from time import sleep
import TimeTagger
from datetime import datetime

tagger = TimeTagger.createTimeTagger(resolution=TimeTagger.Resolution.Standard)
tagger.setDeadtime(1, 2)
serial = tagger.getSerial()
print(f'Connected to time tagger {serial}')
tagger.setTriggerLevel(1, 0.2)
tagger.sync()

sleep(2)

current_time = datetime.now()
print('Current time: ', current_time)
formatted_time = current_time.strftime("%H-%M-%S")
window_size_secs = 50E-3
window_size_ps = window_size_secs*1E12 # Window time in ps
# filewriter = TimeTagger.FileWriter(tagger, f'calibration_msg_15218_symbols_128_samples_per_slot_{CALIBRATION_SYMBOL}_CCSDS_ASM_{formatted_time}', channels=[1])
filewriter = TimeTagger.FileWriter(tagger, f'jupiter_tiny_greyscale_64_samples_per_slot_CSM_{db_att}_interleaved_{formatted_time}', channels=[1])
filewriter.startFor(int(window_size_ps), clear=True)
filewriter.waitUntilFinished()

num_events = filewriter.getTotalEvents()

print(f'{num_events} events written to disk. ')