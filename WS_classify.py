import sys
from WS_train import *

class Classifier:
    
    def __init__(self):
        self.network = NeuralNetwork()
    
    

if __name__ == "__main__":
    # Get filename from arguments
    if len(sys.argv) != 2:
        print "Incorrect usage. Please provide filename or path as argument: " +\
        "\npython WhatSound_download.py <keyword>"
        sys.exit()
    
    filepath = sys.argv[1]
    
    c = Classifier()
    c.network.test_on_file(filepath, 2)
    
        
    
