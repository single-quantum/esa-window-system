import os
import re
import pandas as pd
import decode_from_time_tagger
import matplotlib.pyplot as plt
import numpy as np

def LoadData(path_input,overwrite_file=False,generate=False):
    ppm_orders={}
    last_ppm_number=-1
    for (dir_path, dir_names, file_names) in os.walk(path_input,topdown=True):
        current_dir=os.path.split(dir_path)[-1]
        print(current_dir)
        temp=re.findall(r'\d+',current_dir)
        if(len(temp)>0):
            number=int(temp[0])
        else:
            number=-1
        print(number)
        for val in file_names:
            if(os.path.splitext(val)[-1]=='.xlsx'):
                temp=os.path.join(dir_path,val)
                last_ppm_number=number
                excel_data=pd.read_excel(temp,sheet_name=None)['Sheet1']
                dat=excel_data.set_index('Total attenuation').T.to_dict('list')
                print(dat)
                ppm_orders[number]=dat
            elif('metadata' in val):
                temp=os.path.join(dir_path,val)
                DEBUG_MODE=False
                decode_from_time_tagger.DEBUG_MODE=False
                if(generate):
                    for i in range(1,5):
                        output_data_filename='output_'+str(i)+'_channel_B'
                        time_tagger_channels = np.arange(i)
                        print(os.path.join(dir_path,output_data_filename),os.path.isfile(os.path.join(dir_path,output_data_filename)))
                        if((not os.path.isfile(os.path.join(dir_path,output_data_filename))) or overwrite_file):
                            try:
                                time_events, metadata=decode_from_time_tagger.load_timetagger_data(True,True,dir_path,time_tagger_channels)
                                data_for_analysis=decode_from_time_tagger.analyze_data(time_events,metadata)
                                data_for_analysis['time_tagger_channels']=time_tagger_channels
                                print('decoding with '+str(i)+'channels')
                                decode_from_time_tagger.save_data(data_for_analysis,dir_path,output_data_filename)
                            except:
                                print('Cant decode'+temp)
                        else:
                            print('Skipped creating file'+output_data_filename)
                for i in range(1,5):
                    try:
                        dat=decode_from_time_tagger.load_output_data(dir_path,'output_'+str(i)+'_channel_B')
                        ppm_orders[last_ppm_number][number].append(dat)
                        print('fixed')
                    except:
                        print('Cant load'+temp)
               
    
               
    return ppm_orders

if __name__ == '__main__':
    data=LoadData('C:/Users/SQ/Documents/Dev/esa-window-system/experimental results/15-12-2023/16 ppm',overwrite_file=False,generate=False)
    print(data)
    for i in range(1,5):
        
        total_attenuation=[]
        ref_power=[]
        counts=[]
        counts_TT=[]
        y=[]
        y2=[]
        for key,value in data.items():
            for key2, value2 in value.items():
                
                for j in range(4,len(value2)):
                    if(np.all(value2[j]['time_tagger_channels']==np.arange(i))):
                # if(value2):
                        
                        total_attenuation.append(key2)
                        y.append(value2[j].get('BER before decoding'))
                        y2.append(value2[j].get('BER after decoding'))
                        ref_power.append(value2[1])
                        counts.append(value2[1]*value2[3]/22.7e6)
                        counts_TT.append(value2[2])
        #plt.plot(counts,y,'.-',label='before decoding, number of channels='+str(i))
        plt.plot(counts,y2,'.-',label='after decoding, number of channels='+str(i))
        #plt.plot(x,y2,label='after decoding, number of channels='+str(i))
    plt.xlabel('Photons per pulse')
    plt.ylabel('BER')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()
    print('Done')