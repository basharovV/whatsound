# WhatSound global parameters
#
# These globals determine the parameters for the structure and functionality of 
# the audio classification library
#
# Vyacheslav Basharov. Hons Project 2016"""

global N_hidden_layers
global Learning_rate
global N_output
global N_input
global sigmoid_function
global weights_hid_out
global weights_in_hid
global weights_in_out
global features_saved
global feature_list
global debug

debug = True
debug_extra = True

N_output = 4
N_input = 28
N_hidden_nodes = 14
Learning_rate = 0.05
Momentum = 0.2
Weight_decay = 0.00001

"""
Relative paths for the testing and training data sets.

The required directory structure is:

../samples
    /train/
        /music/*
        /voice/*
        /misc/*
        /silence/*
    /test/
        /music/*
        /voice/*
        /misc/*
        /silence/*
        
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

# Feature extraction parameters
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
