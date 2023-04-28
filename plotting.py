import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from ppm_parameters import symbol_length, slot_length

def plot_timing_corrections(timing_corrections, symbol_length, slot_length, subtitle, twiny_ticks=True):
    symbol_ticks = np.arange(0, len(timing_corrections), 2000)
    time_ticks = np.round(np.array(symbol_ticks)*symbol_length*1E3, 2)
    time_correction_ticks = np.flip(np.round(np.arange(0, timing_corrections[-1]-20E-9, 20E-9)*1E9, 1))
    time_correction_bins = [i*slot_length*1E9 for i in range(1, len(time_correction_ticks)+1)]
    time_correction_bins_tick_labels = [str(i) for i in range(1, len(time_correction_ticks)+1)]


    # Plot timing corrections
    fig, ax = plt.subplots(figsize=(9, 8))


    ax_t = ax.secondary_xaxis('top')
    ax_r = ax.secondary_yaxis('right')

    # ax_r.set_yticklabels()

    ax.set_xlabel('Symbol number')
    ax.set_ylabel('Time correction (ns)')

    ax_t.set_xlabel('Time (ms)')

    ax.plot(np.array(timing_corrections)*1E9)
    # ax.hlines(slot_length*1E9, 0, len(timing_corrections), color='r', linestyles='--', linewidth=1)
    # ax.hlines(8*slot_length*1E9, 0, len(timing_corrections), color='r', linestyles='--', linewidth=1)

    # ax.set_xticks(symbol_ticks)
    # ax.set_xticklabels(symbol_ticks)
    # ax.set_yticks(time_correction_ticks)
    # ax.set_yticklabels(time_correction_ticks)
    ax.set_xlim(0, len(timing_corrections))


    # ax_t.set_xticklabels(time_ticks)

    # if twiny_ticks:
    #     ax_r.set_yticks(time_correction_bins)
    #     ax_r.set_yticklabels(time_correction_bins_tick_labels)
    #     ax_r.set_ylabel('Time correction (# bins)')

    # figs = list(map(plt.figure, plt.get_fignums()))   
    # for fg in figs:
    #     fg.canvas.manager.window.move(1400,-1000)
    plt.grid(alpha=0.7)
    plt.title(f'Time correction per symbol ({subtitle})')
    plt.show()

def plot_hello_world_message():
    with open('pickle files/Hello_world_no_asm_xdata', 'rb') as f:
        x_data = pickle.load(f)
    
    with open('pickle files/Hello_world_no_asm_ydata', 'rb') as f:
        y_data = pickle.load(f)
    
    # convert x_data to microseconds
    x_data = x_data*1E6
    plt.figure()
    plt.plot(x_data, y_data[:len(x_data)])
    plt.title('"Hello World" PPM modulated')
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel('Voltage (V)')
    plt.tight_layout()
    plt.show()

    # Plot the first 4 symbols
    peaks = find_peaks(y_data, height=0.2, width=10)[0]

    num_symbols = 5
    margin = 20

    num_samples_per_slot: int = 32          # Number of samples per bin
    M: int = 16
    bin_factor: float = 5/4
    num_bins_per_symbol: int = int(bin_factor*M)
    sample_size_awg: float = 1/8.84736E9*1E12
    slot_length: float = sample_size_awg*1E-12*num_samples_per_slot # Length of 1 bin in time
    symbol_length: float = slot_length*num_bins_per_symbol          # Length of 1 symbol in time
    samples_per_symbol = num_samples_per_slot*num_bins_per_symbol

    ymin = -0.05
    ymax = 0.55

    symbol_timestamps = x_data[peaks[0]-margin:peaks[num_symbols+1]-margin]
    symbol_y_data = y_data[peaks[0]-margin:peaks[num_symbols+1]-margin]

    fig, axs = plt.subplots(num_symbols, 1, sharex=True, sharey=True)
    xticks = np.arange(0, num_bins_per_symbol*slot_length*1E6, slot_length*1E6)
    xlabels = [str(i) for i in range(len(xticks))]
    for i in range(num_symbols):
        start_idx = np.where((x_data >= x_data[peaks[0]]+i*symbol_length*1E6) & (x_data <= x_data[peaks[0]]+i*symbol_length*1E6 + slot_length*1E6))[0][0]
        stop_idx = np.where((x_data >= x_data[peaks[0]]+i*symbol_length*1E6) & (x_data <= x_data[peaks[0]]+(i+1)*symbol_length*1E6))[0][-1]
        
        # Shift the start index by a little bit to have a bit of margin around the start
        start_idx -= 10
        stop_idx -= 10
        
        x0 = x_data[start_idx]
        
        axs[i].plot(x_data[start_idx:stop_idx]-x0, y_data[start_idx:stop_idx])

        axs[i].vlines(x_data[start_idx]-x0, ymin, ymax, linestyles='--', color='r', linewidth=1)
        # Make the guard slots shaded
        axs[i].axvspan(x_data[start_idx]-x0+16*slot_length*1E6, x_data[start_idx]-x0+20*slot_length*1E6, color='grey', alpha=0.5)
        for k in range(1, 16):
            axs[i].vlines(x_data[start_idx]-x0+k*slot_length*1E6, ymin, ymax, linestyles='--', color='g', linewidth=1)
        
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels(xlabels)
        fig.text(x=0.93, y=-0.235+(1-i*0.16), s=f'Symbol {i}', ha='center', rotation=90)
        # axs[i].set_xlabel('Slot number')
        # axs[i].set_ylabel('Voltage (V)')
        

    plt.suptitle('First 5 symbols of a "Hello World" PPM message', y = 0.93, va='center')
    fig.text(0.5, 0.03, 'Slot number', ha='center')
    fig.text(0.04, 0.5, 'Voltage (V)', va='center', rotation='vertical')
    plt.ylim(ymin, ymax)
    plt.show()

    plt.figure()
    plt.plot(symbol_timestamps, symbol_y_data)
    
    plt.title('"Hello World" PPM modulated')
    plt.xlabel(r'Time ($\mu$s)')
    plt.ylabel('Voltage (V)')
    plt.ylim(ymin, ymax)

    # plt.plot(x_data[peaks[0]-margin:peaks[3]-margin], y_data[peaks[0]-margin:peaks[3]-margin])
    
    for i in range(num_symbols+1):
        plt.vlines(x_data[peaks[0]]+i*symbol_length*1E6, ymin, ymax, linestyles='--', color='r', linewidth=1)
        for j in range(1, 16):
            plt.vlines(x_data[peaks[0]]+i*symbol_length*1E6+j*slot_length*1E6, ymin, ymax, linestyles='--', color='g', linewidth=1)
    
    plt.vlines(x_data[peaks[0]]+(i+1)*symbol_length*1E6, ymin, ymax, linestyles='--', color='r', linewidth=1)
    
    plt.show()

    print()

if __name__ == '__main__':
    plot_hello_world_message()

# %%
import pickle
with open('ppm_messages/ppm_message_Hello_World_no_ASM.csv', 'r') as f:
    lines = f.readlines()
    lines = [int(float(l.replace('\n', ''))) for l in lines]