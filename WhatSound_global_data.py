
global N_hidden_layers
global Learning_rate
global N_output
global N_input
global sigmoid_function
global weights_hid_out
global weights_in_hid
global weights_in_out


################# CHANGE NEURAL NETWORK GLOBAL VARIABLES HERE: ###############


# Number of output neurons
N_output = 3
# Number of input neurons
N_input = 13
# Number of hidden layers
N_hidden_layers = 1
# Number of hidden neurons
N_hidden_neurons = int(N_input / 2)
# Type of sigmoid function: 
#   1 for Hyperbolic Tangent
#   0 for Logistic function
sigmoid_function = 0
# Alpha - learning rate
Learning_rate = 0.1


############### WEIGHT ARRAYS #######################
