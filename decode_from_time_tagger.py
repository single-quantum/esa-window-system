import pickle
import re
from pathlib import Path
import os

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


def get_time_events_from_tt_file(time_events_filename: str, num_channels: int,
                                 get_time_events_per_channel=True, **kwargs):
    """Open the `time_events_filename` with the TimeTagger.FileReader class and retrieve events.

    Can either read out the entire buffer or read out a given number of events. """
    print(time_events_filename)
    time_events_filename2=str(time_events_filename).replace(os.sep,'/')
    print(time_events_filename2)
    print(str(time_events_filename))
    print('c:/Users/SQ/Documents/Dev/esa-window-system/experimental results/15-12-2023/32 ppm/41 dBm (15)/herbig-haro-211-small_10-sps_32-PPM_2-3-code-rate_14-45-55_1702647955.1.ttbin')
    fr = TimeTagger.FileReader(time_events_filename2)


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

        if get_time_events_per_channel:
            channels = data.getChannels()

            events_per_channel = [list(filter(lambda e: e[1] == i, zip(events, channels)))
                                  for i in range(1, num_channels + 1)]
            for i in range(num_channels):
                time_stamps_per_channel[i].append(list(map(lambda e: e[0], events_per_channel[i])))

        time_stamps.append(events)

    for i in range(num_channels):
        time_stamps_per_channel[i] = flatten(time_stamps_per_channel[i])

    time_stamps = flatten(time_stamps)
    # Time stamps from the time tagger are in picoseconds, but the rest of the code uses seconds as the base unit
    time_stamps = np.array(time_stamps)
    # time_events_per_channel = np.zeros()
    # time_events_per_channel = np.array(time_stamps_per_channel)
    time_events = time_stamps * 1E-12

    return time_events, time_stamps_per_channel


def load_timetagger_data(use_latest_tt_file: bool, GET_TIME_EVENTS_PER_SECOND : bool, time_tagger_files_dir: str, time_tagger_channels):
    time_tagger_files_path: Path = Path(__file__).parent.absolute() / time_tagger_files_dir
    tt_files = time_tagger_files_path.rglob('*.ttbin')

    time_tagger_filename: str | Path

    # You can choose to manually put in the time tagger filename below, or use the last added file to the directory.
    if not use_latest_tt_file:
        metadata_filename = 'timetags_metadata_1702647955'
        metadata_id = metadata_filename.split('_')[-1]
        metadata_filepath = time_tagger_files_dir / Path(metadata_filename)
        time_tagger_files = list(filter(lambda f: metadata_id in f.name, tt_files))
        print(time_tagger_files)
        time_tagger_filename = time_tagger_files[-1]
        print(time_tagger_filename)
    else:
        files: list[Path] = [x for x in tt_files if x.is_file()]
        files = sorted(files, key=lambda x: x.lstat().st_mtime)
        time_tagger_filename = os.path.join(time_tagger_files_dir,(re.split(r'\.\d{1}', files[-1].stem)[0] + '.ttbin'))
        print('jher')
        print(time_tagger_filename)
        time_tagger_file_epoch = time_tagger_filename.split('_')[-1].rstrip('.ttbin')

        # Get metadata files
        files_list = time_tagger_files_path.rglob('*')
        metadata_files = [f for f in files_list if not f.suffix]
        metadata_filepath = list(filter(lambda f: time_tagger_file_epoch in f.name, metadata_files))[0]

    if not metadata_filepath.exists():
        raise FileNotFoundError('Metadata file not found. Check path / filename.')

    with open(metadata_filepath, 'rb') as f:
        metadata = pickle.load(f)
        

    time_events, time_events_per_channel = get_time_events_from_tt_file(
        time_tagger_filename, 4, get_time_events_per_channel=GET_TIME_EVENTS_PER_SECOND)
    # Remove duplicate timing events
    time_events = np.unique(time_events)
    
    if GET_TIME_EVENTS_PER_SECOND:
        time_events = np.sort(np.concatenate(
            tuple(np.array(time_events_per_channel[i]) for i in time_tagger_channels)) * 1E-12)
        
    return time_events, metadata

def analyze_data(time_events, metadata):
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

    #time_events_samples = (time_events - time_events[0]) * (8.82091E9 / num_samples_per_slot) + 0.5

    

    print(f'Number of events: {len(time_events)}')

    slot_mapped_message, events_per_slot = demodulate(
        time_events,
        M,
        slot_length,
        symbol_length,
        csm_correlation_threshold=CORRELATION_THRESHOLD,
        message_idx=MESSAGE_IDX,
        **{
            'num_samples_per_slot': num_samples_per_slot,
            'debug_mode': DEBUG_MODE
        }
    )

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

    information_blocks, BER_before_decoding = decode(##Problem, this resets the value for BER_before_decoding!
        slot_mapped_message, M, CODE_RATE,
        use_inner_encoder=USE_INNER_ENCODER,
        **{
            'use_cached_trellis': False,
            'num_events_per_slot': events_per_slot,
            'use_randomizer': USE_RANDOMIZER,
            'sent_bit_sequence_no_csm': sent_bits_no_csm,
            'sent_bit_sequence': sent_bits
        })

    
    print('BER before decoding', BER_before_decoding)
    BER_before_decoding = np.sum([abs(x - y) for x, y in zip(received_bits, sent_bits)]) / len(sent_bits)
    
    sent_img_array: npt.NDArray[np.int_] = np.array([])
    img_arr: npt.NDArray[np.int_] = np.array([])
    CMAP = ''

    sent_message: npt.NDArray = np.array([])

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
    

    if DEBUG_MODE and PAYLOAD_TYPE == 'image':
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
    
    data_for_analysis={}
    data_for_analysis['BER before decoding']=BER_before_decoding
    data_for_analysis['BER after decoding']=BER_after_decoding

    print('Analysis done')
    return data_for_analysis

def save_data(data_to_be_saved,path,name='output_data'):
    
    with open(os.path.join(path,name), 'wb') as handle:
        pickle.dump(data_to_be_saved, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Saving done')

def load_output_data(path,name='output_data'):
    data={}
    with open(os.path.join(path,name), 'rb') as handle:
        data=pickle.load(handle)
    return data

if __name__ == '__main__':
    DEBUG_MODE=True
    use_latest_tt_file: bool = True
    GET_TIME_EVENTS_PER_SECOND = True
    time_tagger_files_dir: str = 'C:/Users/SQ/Documents/Dev/esa-window-system/experimental results/15-12-2023/16 ppm/42 dBm (16)/'
    # time_tagger_files_dir: str = 'time tagger files/'
    time_tagger_channels = [0,1]

    time_events, metadata=load_timetagger_data(use_latest_tt_file,GET_TIME_EVENTS_PER_SECOND,time_tagger_files_dir,time_tagger_channels)
    print('here')
    data_for_analysis=analyze_data(time_events,metadata)
    save_data(data_for_analysis,time_tagger_files_dir)
    print(load_output_data(time_tagger_files_dir))
    
    # reference_file_path = f'jupiter_greyscale_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence'
