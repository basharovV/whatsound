import sys
import pyaudio
import argparse
import numpy as np
import essentia
import atexit
import threading as th
import time
import WS_utils
from WS_train import *

class Classifier:
    
    def __init__(self):
        self.network = NeuralNetwork()
    
    def classify_file(filepath):
        self.network.test_on_file(filepath, 2)

class StreamClassifier(Classifier):
    
    def __init__(self, buffer_size = WS_global_data.buffer_size, 
                channels = WS_global_data.channels, 
                sample_rate = WS_global_data.sample_rate):
        Classifier.__init__(self)
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.channels = channels        
        self.format = pyaudio.paInt32
        # self.lock = th.Lock()
        self.paudio = pyaudio.PyAudio()
        self.stream = self.paudio.open(format=self.format, 
                                  channels = self.channels, 
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer = self.buffer_size)
        self.frames = []                          
        self.buffer = np.array([])
        atexit.register(self.stop_listening)
        
    def __stream_callback(self, in_data, frame_count, time_info, status):
        self.frames.append(in_data)
        # Convert from string audio representation to float numpy array
        frames_float = np.fromstring(in_data, 'Int32')
        # Horizontal stack - convert nd array to 1d array by stacking columns
        self.buffer = np.hstack((self.buffer, frames_float))
        return None, pyaudio.paContinue
    
    def record(self, seconds):
        self.clear()
        for i in range(0, int(self.sample_rate / self.buffer_size * seconds)):            
            frame_raw = self.stream.read(self.buffer_size)
            frame_audio = np.fromstring(frame_raw, 'Int32')
            self.frames.append(frame_raw)
            self.buffer = np.hstack((self.buffer, frame_audio))
        return (self.buffer, self.frames)
            
    def clear(self):
        self.buffer = np.array([])         
        self.frames = []
    
    def start_listening(self):
        self.stream.start_stream()
        
    def stop_listening(self):
        self.stream.stop_stream()
    
    def get_buffer(self):
        with self.lock:
            result_buffer = self.buffer
            result_raw = self.frames
            self.buffer = np.array([])
            self.frames = []
            return (result_buffer, result_raw)
            
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
SIZE = 16

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
    
    print("* recording (3 seconds)")
    frames = []
    audio = np.array([])
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        floats = np.fromstring(data, 'Float32')
        frames.append(data)
        audio = np.hstack((audio,floats))
    print("* done")
    stream.stop_stream()
    stream.close()
    p.terminate()
    WS_utils.write_audio_signal(frames,'recording.wav')
    return audio

if __name__ == "__main__":
    # Get filename from arguments
    args = sys.argv[1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="The path of the file to classify")
    parser.add_argument("-r", "--realtime", help="Enable realtime classification",
                        action="store_true")
    
    args = parser.parse_args()
    if (not (args.file or args.realtime)):
        print "Incorrect usage " 
        parser.print_help()
        sys.exit(0)
    
    print "\n\n. . . . . . . . . . . WhatSound classifier . . . . . . . . . . . \n"
    if args.file:
        print "Classifier mode : file"     
        filepath = args.file   
        fc = Classifier()
        fc.network.test_on_file(filepath, 2)
        
    elif args.realtime:
        print "Classifier mode : [real-time]"        
        print "Initialising PyAudio...\n"
        sc = StreamClassifier()        
        sc.start_listening()
        # WS_utils.clear_console()
        print "\nInitialisation finished. \nListening...\n"
    
        while(1):            
            audio, frames = sc.record(WS_global_data.record_length)   
            # Currently testing on a .wav file recorded every x seconds
            # The raw audio signal doesn't seem to map well to the audio signal
            # read from a file in Essentia, so the feature values are all in the
            # wrong range. 
            # Need to look for a fix, as reading in the array is much faster than
            # a file.  
            WS_utils.write_audio_signal(frames, '../realtime/testfile.wav')
            # sc.network.test_on_signal(audio)
            class_label = sc.network.test_on_file('../realtime/testfile.wav' , verbose=False)
            sys.stdout.write("\rAudio class:  (*****    %s    *****)" % class_label)
            sys.stdout.flush()
    
            
    
    
        
    
