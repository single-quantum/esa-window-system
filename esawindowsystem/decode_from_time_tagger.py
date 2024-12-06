import os
import pickle
import re
from pathlib import Path
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import TimeTagger
import matplotlib as mpl

from esawindowsystem.core.BCJR_decoder_functions import ppm_symbols_to_bit_array
from esawindowsystem.core.data_converter import payload_to_bit_sequence
from esawindowsystem.core.demodulation_functions import demodulate
from esawindowsystem.core.encoder_functions import map_PPM_symbols, get_asm_bit_arr
from esawindowsystem.core.scppm_decoder import decode
from esawindowsystem.core.utils import flatten, calculate_num_photons
from esawindowsystem.ppm_parameters import (CORRELATION_THRESHOLD, DEBUG_MODE, MESSAGE_IDX,
                                            USE_INNER_ENCODER, USE_RANDOMIZER)

"""Read time tagger files from the Swabian Time Tagger Ultra. Required software for the time tagger can be found here:
https://www.swabianinstruments.com/time-tagger/downloads/ . """


def get_time_events_from_tt_file(time_events_filename: str | Path, num_channels: int,
                                 get_time_events_per_channel=True, **kwargs):
    """Open the `time_events_filename` with the TimeTagger.FileReader class and retrieve events.

    Can either read out the entire buffer or read out a given number of events. """
    print(time_events_filename)
    time_events_filename2 = str(time_events_filename).replace(os.sep, '/')
    print(time_events_filename2)
    print(str(time_events_filename))

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


def load_timetagger_data(use_latest_tt_file: bool, GET_TIME_EVENTS_PER_SECOND: bool,
                         time_tagger_files_dir: str, time_tagger_channels, calibrate_time_tags: bool = False):
    time_tagger_files_path: Path = Path(__file__).parent.absolute() / time_tagger_files_dir
    tt_files = time_tagger_files_path.rglob('*.ttbin')

    time_tagger_filename: str | Path

    # You can choose to manually put in the time tagger filename below, or use the last added file to the directory.
    if not use_latest_tt_file:
        metadata_filename = 'timetags_metadata_1730297682'
        metadata_id = metadata_filename.split('_')[-1]
        metadata_filepath = time_tagger_files_dir / Path(metadata_filename)
        time_tagger_files = list(filter(lambda f: metadata_id in f.name, tt_files))
        print(time_tagger_files)
        time_tagger_filename = time_tagger_files[-1]
        print(time_tagger_filename)
    else:
        files: list[Path] = [x for x in tt_files if x.is_file()]
        files = sorted(files, key=lambda x: x.lstat().st_mtime)
        if not files:
            raise IndexError("Time tagger directory empty. Make sure path is correct. \n" +
                             f"Current path: {time_tagger_files_path}")
        time_tagger_filename = os.path.join(time_tagger_files_dir, (re.split(r'\.\d{1}', files[-1].stem)[0] + '.ttbin'))
        print(time_tagger_filename)
        time_tagger_file_epoch = time_tagger_filename.split('_')[-1].rstrip('.ttbin')

        # Get metadata files
        files_list = time_tagger_files_path.rglob('*')
        metadata_files = [f for f in files_list if 'metadata' in f.stem]
        metadata_filepath = list(filter(lambda f: time_tagger_file_epoch in f.name, metadata_files))[0]

    if not metadata_filepath.exists():
        raise FileNotFoundError('Metadata file not found. Check path / filename.')

    with open(metadata_filepath, 'rb') as f:
        metadata = pickle.load(f)

    time_events, time_events_per_channel = get_time_events_from_tt_file(
        time_tagger_filename, 4, get_time_events_per_channel=GET_TIME_EVENTS_PER_SECOND)
    # Remove duplicate timing events
    time_events = np.unique(time_events)

    if calibrate_time_tags:
        for i in range(len(time_events_per_channel[1])):
            time_events_per_channel[1][i] += 46
        for i in range(len(time_events_per_channel[2])):
            time_events_per_channel[2][i] += 96
        for i in range(len(time_events_per_channel[3])):
            time_events_per_channel[3][i] += 58

    if GET_TIME_EVENTS_PER_SECOND:
        time_events = np.sort(np.concatenate(
            tuple(np.array(time_events_per_channel[i]) for i in time_tagger_channels)) * 1E-12)

    # if calibrate_time_tags:
    #     for channel in [1, 2, 3]:
    #         hist = []
    #         limit = 15000
    #         for t_start in time_events_per_channel[0][:30000]:
    #             for t_stop in time_events_per_channel[channel][:30000]:
    #                 if t_start > t_stop:
    #                     continue
    #                 dt = t_stop - t_start
    #                 if dt > limit:
    #                     break
    #                 if dt == 0:
    #                     continue
    #                 hist.append(dt)
    #         hist = np.array(hist)

    #         plt.figure()
    #         bin_values, bin_times, _ = plt.hist(hist, bins=300)
    #         plt.show()

    #         max_bin_index = np.argmax(bin_values)
    #         time_shift = bin_times[max_bin_index]

    #         print(f'Relative time shift (channel = {channel}): ', time_shift)

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
    sent_bits = metadata.get('sent_bit_sequence')
    sent_bits_no_csm = metadata.get('sent_bit_sequence_no_csm')
    sent_symbols = metadata.get('sent_symbols')

    if sent_symbols is None:
        sent_symbols = map_PPM_symbols(list(sent_bits), int(np.log2(M)))
    num_slots_per_symbol = int(5 / 4 * M)

    # time_events_samples = (time_events - time_events[0]) * (8.82091E9 / num_samples_per_slot) + 0.5

    print(f'Number of events: {len(time_events)}')

    slot_mapped_message, events_per_slot, estimated_num_photons_per_pulse = demodulate(
        time_events,
        M,
        slot_length,
        symbol_length,
        sent_symbols,
        csm_correlation_threshold=CORRELATION_THRESHOLD,
        **{
            'num_samples_per_slot': num_samples_per_slot,
            'debug_mode': DEMODULATOR_DEBUG_MODE,
        }
    )

    m = int(np.log2(M))
    received_ppm_symbols = np.nonzero(slot_mapped_message)[1]
    received_bits = ppm_symbols_to_bit_array(received_ppm_symbols, m)

    with open('received_bit_sequence', 'wb') as f:
        pickle.dump(received_bits, f)

    if sent_bits is None:
        with open('sent_bit_sequence', 'rb') as f:
            sent_bits = pickle.load(f)

    if sent_bits_no_csm is None:
        with open('sent_bit_sequence_no_csm', 'rb') as f:
            sent_bits_no_csm = pickle.load(f)

    BER_before_decoding = np.sum([abs(x - y) for x, y in zip(received_bits, sent_bits)]) / len(sent_bits)
    print('BER before decoding', BER_before_decoding)

    sent_img_array: npt.NDArray[np.int_] = np.array([])
    img_arr: npt.NDArray[np.int_] = np.array([])
    CMAP = ''

    sent_message: npt.NDArray = np.array([])

    if PAYLOAD_TYPE == 'image':
        # compare to original image
        sent_message = payload_to_bit_sequence(PAYLOAD_TYPE, filepath=IMG_FILE_PATH)

        if GREYSCALE:
            sent_img_array = map_PPM_symbols(sent_message, 8)
            CMAP = 'Greys'
        else:
            CMAP = 'binary'

    information_blocks, BER_before_decoding, where_asms = decode(  # Problem, this resets the value for BER_before_decoding!
        slot_mapped_message, M, CODE_RATE,
        use_inner_encoder=USE_INNER_ENCODER,
        **{
            'use_cached_trellis': False,
            'num_events_per_slot': events_per_slot,
            'use_randomizer': USE_RANDOMIZER,
            'sent_bit_sequence_no_csm': sent_bits_no_csm,
            'debug_mode': DECODER_DEBUG_MODE,
            'sent_bit_sequence': sent_bits
        })

    print('BER before decoding', BER_before_decoding)
    BER_before_decoding = np.sum([abs(x - y) for x, y in zip(received_bits, sent_bits)]) / len(sent_bits)

    # In the case of a greyscale image, each pixel has a value from 0 to 255.
    # This would be the same as saying that each pixel is a symbol, which should be mapped to an 8 bit sequence.
    # if GREYSCALE:
    #     sent_message = ppm_symbols_to_bit_array(sent_img_array.flatten(), 8)
    # else:
    #     sent_message = sent_img_array.flatten()

    transfer_frames = []
    BER_per_transfer_frame = []
    for asm_idx in where_asms:

        information_block_sizes = {
            Fraction(1, 3): 5040,
            Fraction(1, 2): 7560,
            Fraction(2, 3): 10080
        }

        num_bits = information_block_sizes[CODE_RATE]
        ASM_arr = get_asm_bit_arr()

        transfer_frame = information_blocks[asm_idx + ASM_arr.shape[0]:(asm_idx + ASM_arr.shape[0] + num_bits * 8)]
        transfer_frames.append(transfer_frame)

        if len(transfer_frame) < len(sent_message):
            BER_after_decoding = np.sum(np.abs(transfer_frame -
                                        sent_message[:len(transfer_frame)])) / len(transfer_frame)
        else:
            BER_after_decoding = np.sum(
                np.abs(transfer_frame[:len(sent_message)] - sent_message)) / len(sent_message)

        BER_per_transfer_frame.append(BER_after_decoding)

    best_message_idx: int = np.argmin(BER_per_transfer_frame)
    best_message = transfer_frames[best_message_idx]

    if GREYSCALE:
        img_arr = map_PPM_symbols(best_message, 8)
        img_arr = img_arr[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)
    else:
        img_arr = best_message.flatten()[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)

    median_BER = np.median(BER_per_transfer_frame)
    min_BER = BER_per_transfer_frame[best_message_idx]

    print(f'Best BER after decoding: {min_BER:.3e}. ')
    print(f'Median BER after decoding: {median_BER:.3e}')

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

    data_for_analysis = {}
    data_for_analysis['BER before decoding'] = BER_before_decoding
    data_for_analysis['BER after decoding'] = BER_after_decoding
    data_for_analysis['estimated_num_photons_per_pulse'] = estimated_num_photons_per_pulse
    data_for_analysis['median_BER'] = median_BER
    data_for_analysis['min_BER'] = min_BER
    data_for_analysis['BER_per_transfer_frame'] = BER_per_transfer_frame

    print('Analysis done')
    return data_for_analysis, img_arr


def save_data(data_to_be_saved, path, name='output_data'):

    with open(os.path.join(path, name), 'wb') as handle:
        pickle.dump(data_to_be_saved, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Saving done')


def load_output_data(path, name='output_data'):
    data = {}
    with open(os.path.join(path, name), 'rb') as handle:
        data = pickle.load(handle)
    return data


if __name__ == '__main__':
    DECODER_DEBUG_MODE = True
    DEMODULATOR_DEBUG_MODE = True
    use_latest_tt_file: bool = True
    GET_TIME_EVENTS_PER_SECOND = True
    ANALYZE_DATA = True

    time_tagger_channels = [
        [0, 1, 2, 3]
    ]

    detector = "C824-01"
    M = 16
    slot_length = 1

    bit_error_rates = []
    bit_error_rates_per_transfer_frame = []
    estimated_num_photons_per_pulse = []
    num_photons_per_pulse = []

    # C824-02, 8 ppm, 2 ns slots
    # attenuation_levels = [22, 22.5, 23, 23.5]
    # measured_powers = [2.53E-9, 2.26E-9, 2.01E-9, 1.78E-9]
    # attenuation_levels = [23.5]
    # measured_powers = [1.78E-9]

    # C824-01, 8 ppm, 2 ns slots
    # attenuation_levels: list[float] = [18, 18.5, 19, 19.5, 20]
    # measured_powers = [7.47E-9, 6.70E-9, 6.00E-9, 5.30E-9, 4.70E-9]
    # attenuation_levels: list[float] = [19, 19.5, 20]
    # measured_powers = [6.00E-9, 5.30E-9, 4.70E-9]

    # C824-02, 16 ppm, 500 ps slots
    # measured_powers = [7.09, 6.22, 5.46]
    # attenuation_levels = [19, 19.5, 20]

    # C824-02, 16 ppm, 1 ns slots
    # measured_powers: list[float] = [6.25, 5.56, 4.94, 4.40, 3.91, 3.48]
    # attenuation_levels: list[float] = [21.5, 22, 22.5, 23, 23.5, 24]

    # C824-02, 16 ppm, 2 ns slots
    # measured_powers: list[float] = []
    # attenuation_levels: list[float] = []

    # C824-01, 16 ppm, 500 ps slots
    # measured_powers: list[float] = [27.3, 24.23, 21.52, 19.20, 17.05]
    # attenuation_levels: list[float] = [14, 14.5, 15, 15.5, 16]

    # C824-01, 16 ppm, 1 ns slots
    measured_powers: list[float] = [14.47, 12.90, 11.48, 10.22, 9.02]
    attenuation_levels: list[float] = [18, 18.5, 19, 19.5, 20]
    # measured_powers: list[float] = [11.48, 10.22, 9.02]
    # attenuation_levels: list[float] = [19, 19.5, 20]

    # C824-01, 16 ppm, 2 ns slots
    # measured_powers: list[float] = []
    # attenuation_levels: list[float] = []

    num_pulses_per_second = 44.1E6
    attenuation_levels_str = [str(i).replace('.', '_') for i in attenuation_levels]

    for att_idx, attenuation in enumerate(attenuation_levels_str):
        # time_tagger_files_dir: str = 'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\C824-02\\16 ppm\\1 ns slots'
        # time_tagger_files_dir: str = 'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\C824-02\\16 ppm\\500 ps slots'
        # time_tagger_files_dir: str = 'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\24-10-2024\\Non-Bridged detectors\\C824-08\\21 db attenuation'
        # time_tagger_files_dir: str = 'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\time tagger files\\before 01-11-2024'
        time_tagger_files_dir: str = f'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\{detector}\\{M} ppm\\{slot_length} ns slots\\{
            attenuation} db attenuation'
        # time_tagger_files_dir: str = 'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\time tagger files'
        img_arrs = []

        for i, channels in enumerate(time_tagger_channels):
            time_events, metadata = load_timetagger_data(
                use_latest_tt_file, GET_TIME_EVENTS_PER_SECOND, time_tagger_files_dir, channels, calibrate_time_tags=True)

            # num_photons_per_pulse.append(calculate_num_photons(
            #     measured_powers[att_idx], num_pulses_per_second, detector_efficiency=0.21))

            if ANALYZE_DATA:
                data_for_analysis, img_arr = analyze_data(
                    time_events[:200000], metadata)
                img_arrs.append(img_arr)

                estimated_num_photons_per_pulse = data_for_analysis['estimated_num_photons_per_pulse']
                BER_after_decoding = data_for_analysis['min_BER']
                bit_error_rates_per_transfer_frame.append(data_for_analysis['BER_per_transfer_frame'])

        IMG_SHAPE = metadata.get('IMG_SHAPE')
        PAYLOAD_TYPE = metadata.get('PAYLOAD_TYPE')
        IMG_FILE_PATH = metadata.get('IMG_FILE_PATH')
        sent_message = payload_to_bit_sequence(PAYLOAD_TYPE, filepath=IMG_FILE_PATH)
        sent_img_array = map_PPM_symbols(sent_message, 8)
        original_img_arr = sent_img_array[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE)

        # if ANALYZE_DATA:
        #     difference_mask = np.where(original_img_arr != img_arr)
        #     highlighted_image = img_arr.copy()
        #     highlighted_image[difference_mask] = -1

        #     custom_cmap = mpl.colormaps['Greys']
        #     custom_cmap.set_under(color='r')
        #     CMAP = 'Greys'

        #     label_font_size = 14
        #     fig, axs = plt.subplots(1, 2, figsize=(5, 4))
        #     plt.suptitle('SCPPM message comparison', fontsize=18)
        #     axs[0].imshow(original_img_arr, cmap=CMAP)
        #     axs[0].set_xlabel('Pixel number (x)', fontsize=label_font_size)
        #     axs[0].set_ylabel('Pixel number (y)', fontsize=label_font_size)
        #     axs[0].tick_params(axis='both', which='major', labelsize=label_font_size)
        #     axs[0].set_title('Original image', fontsize=16)

        #     axs[1].imshow(highlighted_image, cmap=custom_cmap, vmin=0)
        #     axs[1].set_xlabel('Pixel number (x)', fontsize=label_font_size)
        #     axs[1].set_ylabel('Pixel number (y)', fontsize=label_font_size)
        #     axs[1].tick_params(axis='both', which='major', labelsize=label_font_size)

        #     axs[1].set_title('Decoded image', fontsize=16)
        #     plt.show()
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(original_img_arr, cmap='Greys')
        # axs[1].imshow(img_arr[:IMG_SHAPE[0] * IMG_SHAPE[1]].reshape(IMG_SHAPE))
        # plt.show()

        num_photons_per_pulse.append(estimated_num_photons_per_pulse)
        bit_error_rates.append(BER_after_decoding)

        with open(f'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\ESA Voyage 2050\\Time tagger files\\{detector}\\{M} ppm\\{slot_length} ns slots\\bit error rates', 'wb') as f:
            pickle.dump(
                {
                    'bit_error_rates_per_transfer_frame': bit_error_rates_per_transfer_frame,
                    'bit_error_rates': bit_error_rates,
                    'num_photons_per_pulse': num_photons_per_pulse,
                    'attenuation_levels': attenuation_levels
                }, f)

    plt.figure()
    plt.semilogy(num_photons_per_pulse, bit_error_rates, marker='x')
    plt.xlabel('Attenuation (dB)')
    plt.ylabel('Bit Error Rate (-)')
    plt.show()

    # axs[0].tick_params(
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     left=False,
    #     labelbottom=False,
    #     labelleft=False
    # )  # labels along the bottom edge are off
    # for i, img_arr in enumerate(decoded_imgs):
    #     axs[i+1].imshow(img_arr, cmap='Greys')
    #     axs[i+1].tick_params(
    #         which='both',      # both major and minor ticks are affected
    #         bottom=False,      # ticks along the bottom edge are off
    #         top=False,         # ticks along the top edge are off
    #         left=False,
    #         labelbottom=False,
    #         labelleft=False
    #     )  # labels along the bottom edge are off

    print('done')
    # save_data(data_for_analysis, time_tagger_files_dir)
    # print(load_output_data(time_tagger_files_dir))

    # reference_file_path = f'jupiter_greyscale_{num_samples_per_slot}_samples_per_slot_{M}-PPM_interleaved_sent_bit_sequence'
