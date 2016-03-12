import wave
import WS_global_data
import pyaudio
import os
import numpy as np
from pybrain.datasets import ClassificationDataSet
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
    
def supervised_to_classification(dataset, **kwargs):
    hid_nodes = kwargs.pop('hidden_nodes') if kwargs.has_key('hidden_nodes') \
        else WS_global_data.N_hidden_nodes
    classification_set = ClassificationDataSet(WS_global_data.N_input,
                                           nb_classes=WS_global_data.N_output,
                                           class_labels=WS_global_data.Classes)
    for i in xrange(0, dataset.getLength()):
        target_class = np.argmax(dataset.getSample(i)[1])
        classification_set.addSample( dataset.getSample(i)[0], target_class)
    
    return classification_set

def get_set_freqs(dataset):
    frequencies = [0 for i in range(WS_global_data.N_output)]
    for i in xrange(len(dataset)):
        target_class = np.argmax(dataset.getSample(i)[1])
        frequencies[target_class]+=1
    return frequencies

def get_confusion_matrix(hitrates, total_freqs):
    # Confusion matrix indexing: matrix[predicted class][target class]
    confusion_matrix = np.matrix([[0.0 for i in range(WS_global_data.N_output)] \
                        for i in range(WS_global_data.N_output)])
                        
    for i in xrange(len(hitrates)): # Iterate over target class
        for j in xrange(len(hitrates[i])): # Iterate over predicted class
            # Calculate percentage hitrate
            if (total_freqs[i] != 0):
                confusion_matrix[i][j] = hitrates[i][j] / float(total_freqs[i])
            else:
                confusion_matrix[i][j] = 0
    return confusion_matrix
