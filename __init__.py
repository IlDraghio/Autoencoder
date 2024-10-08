from module import *
import torch
import itertools

audio_data_list = get_data(CHANNELS_NUMBER,FRAME_RATE,SAMPLE_WIDTH)

data_fragmented = []
fragments_counter = 0
for data in audio_data_list:
    fragments, num_fragments = cut_audio(data,FRAGMENT_LENGTH_MS)
    data_fragmented.append(fragments)
    fragments_counter+=num_fragments
print(f"Processed {fragments_counter} audio fragments. Be aware that some audio files's end could be cut off by some frames.")

data_fragmented_flattened = list(itertools.chain.from_iterable(data_fragmented))

data_fragmented = np.array(data_fragmented_flattened)

data -to_pytorch(data_fragmented)