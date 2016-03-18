from __future__ import absolute_import
from whatsound.core import WS_network
from sklearn.grid_search import ParameterGrid
import time
import sys

# CONSTANTS
DATASET = "samples/train/" # Run as a module from whatsound_project
BASE_WEIGHTS_PATH = "whatsound/trainer/weights/"
SPLIT = 0.6 # 60% for training, 40% for testing
epochs = 3000

# PARAMETERS FOR SWEEPING
hid_nodes = 14
lrn_rate = 0.05
momentum = 0.2
weight_decay = 0.0001

# Parameter grid - size 8 param values per parameter

param_grid = {
    'hid_nodes': [2, 4, 8, 10, 12, 14, 18, 24],
    'lrn_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8],
    'momentum': [0.01, 0.1, 0.15, 0.2, 0.4, 0.6, 0.8],
    'weight_decay': [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.4]
}

alt_param_grid = {
    'hid_nodes': [4, 8, 12, 18],
    'lrn_rate': [0.0001, 0.01, 0.1, 0.2],
    'momentum': [0.15, 0.2, 0.4, 0.8],
    'weight_decay': [0.00001, 0.01, 0.1, 0.4]
}

grid = ParameterGrid(alt_param_grid)

network = WS_network.NeuralNetwork(lrn_rate=lrn_rate,
                                   weight_decay=weight_decay,
                                   momentum=momentum,
                                   split=SPLIT,
                                   hid_nodes=hid_nodes,
                                   reset=True)

# Always using the same dataset with same split proportion
network.add_set_from_dir(DATASET)
# print list(grid)

# Prepare log file for this session
log_base_path = "whatsound/trainer/logs/"
log_filename = "log_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
log_filepath = log_base_path + log_filename
log_file = open(log_filepath, 'a+')

# Keep a running record of the best average classification accuracy,
# when the autotrainer is finished we can fetch the confusion matrix for the
# params giving the best results.
best_accuracy = 0.0
best_accuracy_idx = 0
best_accuracy_weights = 0

for i in xrange(len(grid)):
    param_set = grid[i]
    sys.stdout.write("\rTraining on param set " + str(i) + " / " + str(len(grid)))
    sys.stdout.flush()

    # Write to log file before beginning training
    timestamp = time.strftime("%Y/%m/%d-%H:%M:%S")
    log_entry = "[" + timestamp + "]" + \
                ": Training on parameter set " + str(i) + "/" + \
                    str(len(grid)) + \
                "\nParameter values: " + str(param_set)
    log_file.write(log_entry)
    # Set new network params
    param_dict= {
        'hid_nodes': param_set['hid_nodes'],
        'lrn_rate': param_set['lrn_rate'],
        'momentum': param_set['momentum'],
        'weight_decay': param_set['weight_decay']
    }
    network.reconfigure_network(**param_dict)

    # Get time stamp in the format 20160311-180423
    filename = time.strftime("%Y%m%d-%H%M%S") + ".xml"
    weights_file = BASE_WEIGHTS_PATH + filename

    # Tell the network where to save the weights after training
    network.set_weights_file(weights_file)
    # Train on new parameter set
    network.train(epochs=epochs, verbose=False)
    # Get results from testing
    accuracy, info = network.test()

    # Update the running record for best Accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_accuracy_idx = i
        best_accuracy_weights = weights_file

    log_entry_result = "\n" + info + ". Accuracy: " + str(accuracy) + "\n\n"
    log_file.write(log_entry_result)
    log_file.flush()

network.set_weights_file(best_accuracy_weights)
done_info = "--------\nAutotrainer finished parameter sweep :)\n"
done_info += "Best parameter set:\n"
done_info += str(grid[best_accuracy_idx])
done_info += "\nWeights file: " + str(best_accuracy_weights)
done_info += ")\n\nConfusion matrix for this param set:"
done_info += "\n" + str(network.test(confusion_matrix=True))
log_file.write(done_info)
log_file.close()
