import os
import re
import pandas as pd
import decode_from_time_tagger
import matplotlib.pyplot as plt
import numpy as np


def load_data(path_input, overwrite_file=False, generate=False, debug_mode=False):
    ppm_orders = {}
    last_ppm_number = -1
    metadata: dict = {}
    time_events = np.array([])
    for (dir_path, dir_names, file_names) in os.walk(path_input, topdown=True):
        sub_directory_str = os.path.split(dir_path)[-1]
        print(f'Sub-directory: {sub_directory_str}')
        ppm_order_match = re.search(r'\d+', sub_directory_str)
        if ppm_order_match is None:
            raise AttributeError("Could not find PPM order from directory name")

        ppm_order = int(ppm_order_match.group())

        for file_name in file_names:
            if (os.path.splitext(file_name)[-1] == '.xlsx'):
                excel_sheet_path = os.path.join(dir_path, file_name)
                last_ppm_number = ppm_order
                excel_data = pd.read_excel(excel_sheet_path, sheet_name=None)['Sheet1']
                dat = excel_data.set_index('Total attenuation').T.to_dict('list')
                print(dat)
                ppm_orders[ppm_order] = dat
            elif ('metadata' in file_name):
                metadata_file_path = os.path.join(dir_path, file_name)
                decode_from_time_tagger.DEBUG_MODE = debug_mode
                if (generate):
                    for i in range(1, 5):
                        output_data_filename = f'output_{i}_channel_B'
                        time_tagger_channels = np.arange(i)
                        print(os.path.join(dir_path, output_data_filename),
                              os.path.isfile(os.path.join(dir_path, output_data_filename)))
                        if ((not os.path.isfile(os.path.join(dir_path, output_data_filename))) or overwrite_file):
                            try:
                                time_events, metadata = decode_from_time_tagger.load_timetagger_data(
                                    True, True, dir_path, time_tagger_channels)
                            except Exception as e:
                                print(f"Can't load file {metadata_file_path}")
                                print(e)
                            try:
                                data_for_analysis = decode_from_time_tagger.analyze_data(time_events, metadata)
                                data_for_analysis['time_tagger_channels'] = time_tagger_channels
                                print(f'decoding with {i} channel(s)')
                                decode_from_time_tagger.save_data(data_for_analysis, dir_path, output_data_filename)
                            except Exception as e:
                                print(f"Can't decode file {metadata_file_path}")
                                print(e)
                        else:
                            print(f'Skipped creating file {output_data_filename}')
                for i in range(1, 5):
                    try:
                        dat = decode_from_time_tagger.load_output_data(dir_path, 'output_'+str(i)+'_channel_B')
                        ppm_orders[last_ppm_number][ppm_order].append(dat)
                    except:
                        print(f'Cant load {metadata_file_path}')

    return ppm_orders


if __name__ == '__main__':
    ppm_order = 16
    sent_pulses_per_second = 44.10e6
    attenuation_range = np.arange(40, 50)

    data = load_data(f'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\15-12-2023\\{ppm_order} ppm',
                     overwrite_file=False, generate=True)
    print(data)

    plot_data = {i: {
        'BER before decoding': [],
        'BER after decoding': [],
        'ref power': [],
        'counts': [],
        'counts TT': []
    } for i in range(1, 5)}

    for attenuation in attenuation_range:
        measurement_data = data[ppm_order][attenuation]

        for i in range(4, len(measurement_data)):
            num_channels = measurement_data[i].get('time_tagger_channels').shape[0]
            plot_data[num_channels]['BER before decoding'].append(measurement_data[i].get('BER before decoding'))
            plot_data[num_channels]['BER after decoding'].append(measurement_data[i].get('BER after decoding'))
            plot_data[num_channels]['ref power'].append(measurement_data[1])
            plot_data[num_channels]['counts TT'].append(measurement_data[2])
            plot_data[num_channels]['counts'].append(measurement_data[1]*measurement_data[3]/sent_pulses_per_second)

        # if len(measurement_data) == 8:

    plt.figure()
    for i in range(1, 5):
        plt.semilogy(plot_data[i]['counts'], plot_data[i]['BER after decoding'],
                     '-x', label=f'# of pixels = {i}')
    plt.title(f'BER after decoding ({ppm_order} PPM)')
    plt.ylabel('Bit Error Ratio (-)', fontsize=12)
    plt.xlabel('Photons per pulse', fontsize=12)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)
    plt.show()
print('Done')
