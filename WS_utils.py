import wave
import WS_global_data
import pyaudio
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# , MaxAbsScaler, StandardScaler

def write_audio_signal(data, filename):
    pa = pyaudio.PyAudio()    
    wave_file = wave.open(filename, 'wb')
    wave_file.setsampwidth(pa.get_sample_size(WS_global_data.sample_format))
    wave_file.setframerate(WS_global_data.sample_rate)
    wave_file.setnchannels(WS_global_data.channels)
    wave_file.writeframes(b''.join(data))
    wave_file.close()
    
def normalize(array):
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(array)
    array = minmax_scaler.transform(array)
    array_norm = np.reshape(array, -1)
    return array_norm

def clear_console():
    os.system('cls' if os.name=='nt' else 'clear')
    
