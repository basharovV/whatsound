import pyaudio
# WhatSound global parameters
#
# These globals determine the parameters for the structure and functionality of
# the audio classification library
#
# Vyacheslav Basharov. Hons Project 2016"""

debug = False
debug_extra = False
debug_files_only = False
debug_dirs_only = True

# ------------------------ Neural Network SETTINGS ----------------------------

# The autotrainer is run as a package module, so all paths must be relative
# to the parent directory of whatsound/
autotrainer_enabled = False
standalone_prefix = "../../" if not autotrainer_enabled else ''
package_xml_prefix = "whatsound/core/xml/" if autotrainer_enabled else "xml/"

N_output = 4
N_input = 27    # Corresponds to the number of feature values (see feature list)
N_hidden_nodes = 8
Learning_rate = 0.01
Momentum = 0.8
Weight_decay = 0.00001

"""
Path where the network weights are stored.
This path does not apply to the autotrainer, which generates new
weights files for every run.
"""
weights_file = "acc79_20160402-110514.xml"
weights_path = package_xml_prefix + weights_file

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
train_path = standalone_prefix + "samples/train/"
test_path = standalone_prefix + "samples/test/"

"""
Data set options

Toggle and change the 'split by proportion' feature. When enabled, only one
data set is provided, which is split into a training and testing set.
"""
split_enabled = True
data_path = train_path
split_proportion = 0.6

"""
The audio classes that WhatSound supports, and their associated indexes (used
for mapping directories to classes)

These may be altered, but require a matching directory structure.
"""

Classes = ['music', 'voice', 'ambient', 'silence']
Class_indexes = {'music': 0, 'voice': 1, 'ambient': 2, 'silence': 3}

# ---------------------- FEATURE EXTRACTION SETTINGS -------------------------

# 22/04/2016 - removed Pitch strength to test with less features
feature_list = ['mfcc',
                'key_strength',
                'spectral_flux',
                'zcr',
                'lpc']

feature_sizes = {'mfcc': 13, \
                'key_strength': 1,\
                'spectral_flux': 1,\
                'zcr': 1,\
                'lpc' : 11}

features_saved = False

features_xml_file = "wsfeatures2.xml"
features_xml_path = package_xml_prefix + features_xml_file

# ----------------------------- AUDIO SETTINGS ------------------------------
sample_rate = 44100
frame_size = 2048
hop_size = 512
sample_format = pyaudio.paFloat32
record_length = 0.5

# ---------------------------  REAL TIME SETTINGS ----------------------------

buffer_size = 2048
channels = 1
seconds = 1
time_interval = 6.0 # in seconds
