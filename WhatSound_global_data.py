
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


################# CHANGE NEURAL NETWORK GLOBAL VARIABLES HERE: ###############

# Number of output neurons
N_output = 4

# Number of input neurons
N_input = 16

# Number of hidden neurons
N_hidden_layers = 8

Learning_rate = 0.05

Momentum = 0.2

Weight_decay = 0.0008

Classes = ['music', 'voice', 'ambient', 'silence']

Class_indexes = {'music': 0, 'voice': 1, 'ambient': 2, 'silence': 3}

feature_list = ['mfcc','key_strength', 'spectral_flux', 'zerocrossingrate']
feature_sizes = {'mfcc': 13, \
                'key_strength': 1,\
                'spectral_flux': 1,\
                'zerocrossingrate': 1}

features_saved = False
