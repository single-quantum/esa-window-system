import os
import re
import pandas as pd
import decode_from_time_tagger
import matplotlib.pyplot as plt
import numpy as np


def LoadData(path_input, overwrite_file=False, generate=False):
    ppm_orders = {}
    last_ppm_number = -1
    metadata: dict = {}
    time_events = np.array([])
    for (dir_path, dir_names, file_names) in os.walk(path_input, topdown=True):
        current_dir = os.path.split(dir_path)[-1]
        print(current_dir)
        temp = re.findall(r'\d+', current_dir)
        if (len(temp) > 0):
            number = int(temp[0])
        else:
            number = -1
        print(number)
        for val in file_names:
            if (os.path.splitext(val)[-1] == '.xlsx'):
                temp = os.path.join(dir_path, val)
                last_ppm_number = number
                excel_data = pd.read_excel(temp, sheet_name=None)['Sheet1']
                dat = excel_data.set_index('Total attenuation').T.to_dict('list')
                print(dat)
                ppm_orders[number] = dat
            elif ('metadata' in val):
                temp = os.path.join(dir_path, val)
                DEBUG_MODE = True
                decode_from_time_tagger.DEBUG_MODE = True
                if (generate):
                    for i in range(1, 5):
                        output_data_filename = 'output_'+str(i)+'_channel_B'
                        time_tagger_channels = np.arange(i)
                        print(os.path.join(dir_path, output_data_filename),
                              os.path.isfile(os.path.join(dir_path, output_data_filename)))
                        if ((not os.path.isfile(os.path.join(dir_path, output_data_filename))) or overwrite_file):
                            try:
                                time_events, metadata = decode_from_time_tagger.load_timetagger_data(
                                    True, True, dir_path, time_tagger_channels)
                            except:
                                print(f"Can't load file {temp}")
                            try:
                                data_for_analysis = decode_from_time_tagger.analyze_data(time_events, metadata)
                                data_for_analysis['time_tagger_channels'] = time_tagger_channels
                                print(f'decoding with {i} channel(s)')
                                decode_from_time_tagger.save_data(data_for_analysis, dir_path, output_data_filename)
                            except Exception as e:
                                print(f"Can't decode file {temp}")
                                print(e)
                        else:
                            print('Skipped creating file'+output_data_filename)
                for i in range(1, 5):
                    try:
                        dat = decode_from_time_tagger.load_output_data(dir_path, 'output_'+str(i)+'_channel_B')
                        ppm_orders[last_ppm_number][number].append(dat)
                        print('fixed')
                    except:
                        print('Cant load'+temp)

    return ppm_orders


if __name__ == '__main__':
    ppm_order = 16
    sent_pulses_per_second = 44.10e6
    attenuation_range = np.arange(40, 50)

    data = LoadData(f'C:\\Users\\hvlot\\OneDrive - Single Quantum\\Documents\\Dev\\esa-window-system\\experimental results\\15-12-2023\\{ppm_order} ppm',
                    overwrite_file=True, generate=True)
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
        plt.semilogy(plot_data[i]['counts'], plot_data[i]['BER before decoding'],
                     '-x', label=f'number of channels = {i}')
    plt.title(f'BER before decoding ({ppm_order} PPM)')
    plt.ylabel('Bit Error Ratio (-)')
    plt.xlabel('Photons per pulse')
    plt.legend()
    plt.show()
print('Done')
