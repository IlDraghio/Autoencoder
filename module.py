from config import *
import numpy as np
import glob
import numpy as np
import sys
from pydub import AudioSegment
import os

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
            
        for file_path in files:
            
            if not os.path.isfile(file_path):
                print(f"Error: File {file_path} is not a file.")
                sys.exit()
            
            audio = AudioSegment.from_file(file_path)
            
            audio = audio.set_channels(CHANNELS_NUMBER).set_frame_rate(FRAME_RATE).set_sample_width(SAMPLE_WIDTH)
                        
            samples = np.array(audio.get_array_of_samples())
            
            samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
            
            audio_data_list.append(samples)
            
    print(f"Processed {len(audio_data_list)} audio files.")
    
    return(audio_data_list)

def cut_audio(audio_segment, FRAGMENT_LENGTH_MS):
    fragments = []

    num_fragments = len(audio_segment) // FRAGMENT_LENGTH_MS
    
    if num_fragments != 0:
        for i in range(num_fragments):
            start_time = i * FRAGMENT_LENGTH_MS
            end_time = start_time + FRAGMENT_LENGTH_MS
            fragment = audio_segment[start_time:end_time]
            fragments.append(fragment)
        
        if len(audio_segment) % FRAGMENT_LENGTH_MS != 0:
            fragments.append(audio_segment[end_time:])              
    else:
        print("Data duration shorter than FRAGMENT_LENGTH_MS. Review your global variables.")
        sys.exit()
    
    return fragments, num_fragments