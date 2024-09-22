from config import *
import numpy as np
import glob
import numpy as np
import sys
from pydub import AudioSegment
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

def data_integrity_check(index,samples):
    if BYPASS_BAD_VALUES:
        nan_indices = np.where(np.isnan(samples))
        inf_indices = np.where(np.isinf(samples))
        if nan_indices[0].size > 0 or inf_indices[0].size > 0:
            samples[nan_indices] = 0
            samples[inf_indices] = 0
            print(f"Audio file number {index+1} bad values fixed.")
    else:
        print(f"Audio file number {index+1} is corrupted. Try to fix it or set BYPASS_BAD_VALUES = true.")
        sys.exit()

def get_data(CHANNELS_NUMBER,FRAME_RATE,SAMPLE_WIDTH):
    
    if not os.path.exists(DIRECTORY):
        print("The data folder does not exist. Please read the README for instructions on how to use the program.")
        sys.exit()
    
    if not os.listdir(DIRECTORY):
        print("The data folder is empty. Please read the README for instructions on how to use the program.")
        sys.exit()
        
    audio_data_list = []
        
    for ext in EXTENSIONS:
        
        files = glob.glob(os.path.join(DIRECTORY, ext))  
            
        for index, file_path in enumerate(files):
            
            if not os.path.isfile(file_path):
                print(f"Error: File {file_path} is not a file.")
                sys.exit()
            
            audio = AudioSegment.from_file(file_path)
            
            audio = audio.set_channels(CHANNELS_NUMBER).set_frame_rate(FRAME_RATE).set_sample_width(SAMPLE_WIDTH)
                        
            samples = np.array(audio.get_array_of_samples())
            
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
        
            data_integrity_check(index,samples)
                
            audio_data_list.append(samples)
            
    print(f"Processed {len(audio_data_list)} audio files.")
    
    return(audio_data_list)

def cut_audio(audio_data, FRAGMENT_LENGTH_MS):
    fragments = []

    num_fragments = len(audio_data) // FRAGMENT_LENGTH_MS
    
    if num_fragments != 0:
        for i in range(num_fragments):
            start_time = i * FRAGMENT_LENGTH_MS
            end_time = start_time + FRAGMENT_LENGTH_MS
            fragment = audio_data[start_time:end_time]
            fragment = np.array(fragment)
            fragments.append(fragment)  
    else:
        print("Data duration shorter than FRAGMENT_LENGTH_MS. Review your global variables.")
        sys.exit()

    fragments = np.array(fragments)
    return fragments, num_fragments

def data_to_pytorch(fragments):
    
    x_train, x_validation = train_test_split(fragments, test_size=0.2, shuffle=True, random_state=0)
    
    x_train_pt = torch.Tensor(x_train).float()
    x_train_pt = x_train_pt.view(x_train_pt.size(),1,x_train_pt[0].size())

    x_validation_pt = torch.Tensor(x_validation).float()
    x_validation_pt = x_validation_pt.view(x_validation_pt.size(),1,3200)
