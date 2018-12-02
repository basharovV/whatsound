# whatsound - a Python ML toolkit for audio classification

*whatsound* is a toolkit for training, and testing audio classification using a neural network.
- Music
- Speech
- Ambient/Noise
- Silence

This toolkit uses [Essentia](http://essentia.upf.edu/documentation/) for audio feature extraction, and [PyBrain](http://pybrain.org/) for the use of a backpropagation neural network for the training and testing of classification.

## How it works

### WS_classify.py

This is the toolkit for classification, which exposes the main classification functionality.


## Source modules

The project is split into modules fit for different purposes. 

### /core

These modules are needed for audio training and classification.

#### WS_extractor.py

Extracts audio features from a stream. The [Essentia](http://essentia.upf.edu/documentation/) library is used for audio analysis.
The features which are used for extraction are:
- MFCC
- Zero crossing rate
- Key strength
- Spectral Flux
- Pitch strength
- LPC

#### WS_utils.py

Utility functions

#### WS_global_data.py

These are global parameters - settings for the neural net, training parameters, audio settings and classifier types.

#### WS_network.py

This module allows training and testing of a data set, with optional the following parameters:
- `weights` : the path to a PyBrain weights XML file
- `dataset`: the path to a directory containing audio samples split by class
- `split`: the ratio with which to split the data set between training/testing

