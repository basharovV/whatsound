import pyaudio
# WhatSound global parameters
#
# These globals determine the parameters for the structure and functionality of 
# the audio classification library
#
# Vyacheslav Basharov. Hons Project 2016"""

debug = False
debug_extra = False

# ------------------------ CLASSIFICATION SETTINGS ----------------------------

N_output = 4
N_input = 28
N_hidden_nodes = 14
Learning_rate = 0.08
Momentum = 0.3
Weight_decay = 0.0001

"""
Relative paths for the testing and training data sets.

The required directory structure is:

../samples
    /train/                         /test/
        /music/*                        /music/*
        /voice/*                        /voice/*
        /misc/*                         /misc/*
        /silence/*                      /silence/*
Note: subfolders can contain other subfolders.
"""
train_path = "../samples/train/"
test_path = "../samples/test/"

"""
Data set options

Toggle and change the 'split by proportion' feature. When enabled, only one 
data set is provided, which is split into a training and testing set. 
"""
split_enabled = False
data_path = train_path
split_proportion = 0.75

"""
The audio classes that WhatSound supports, and their associated indexes (used
for mapping directories to classes)

These may be altered, but require a different directory structure.
"""

Classes = ['music', 'voice', 'ambient', 'silence']
Class_indexes = {'music': 0, 'voice': 1, 'ambient': 2, 'silence': 3}

# ---------------------- FEATURE EXTRACTION SETTINGS -------------------------
feature_list = ['mfcc',
                'key_strength', 
                'spectral_flux', 
                'zcr',
                'pitch_strength',
                'lpc']

feature_sizes = {'mfcc': 13, \
                'key_strength': 1,\
                'spectral_flux': 1,\
                'zcr': 1,\
                'pitch_strength': 1,\
                'lpc' : 11}

features_saved = False

# ----------------------------- AUDIO SETTINGS ------------------------------
sample_rate = 44100
frame_size = 2048
hop_size = 512
sample_format = pyaudio.paFloat32
record_length = 2.0

# ---------------------------  REAL TIME SETTINGS ----------------------------

buffer_size = 2048
channels = 1
seconds = 1
time_interval = 6.0 # in seconds
