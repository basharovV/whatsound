"""
    'whatsound' - A neural network approach for classifying audio
    -------------------------------------------------------------------------
    This module performs training and classification of provided data sets, with
    one of four possible outputs:
     *music
     *speech
     *ambient
     *silence
    The majority of the data used for training is samples fetched from
    freesound.org using the freesound-python API client library.
    -------------------------------------------------------------------------
    Author: Vyacheslav Basharov
    Version 10/12/2015
"""

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer, SigmoidLayer, TanhLayer
from pybrain.tools.customxml import NetworkReader, NetworkWriter
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.preprocessing import normalize
import numpy as np
from numpy import array
import WhatSound_global_data
from WhatSound_extractor import *
import yaml
import essentia
from __builtin__ import file
import os
import sys


class NeuralNetwork():
    
    def __init__(self,
                    in_nodes=WhatSound_global_data.N_input,
                    weight_decay=WhatSound_global_data.Weight_decay,
                    hid_layers=WhatSound_global_data.N_hidden_layers,
                    out_nodes=WhatSound_global_data.N_output,
                    lrn_rate=WhatSound_global_data.Learning_rate,
                    momentum=WhatSound_global_data.Momentum,
                    classes=WhatSound_global_data.Classes,
                    reset=False):
        
        # INITIALISE NETWORK PARAMS
        self.in_nodes = in_nodes
        self.hid_layers = hid_layers
        self.out_nodes = out_nodes
        self.lrn_rate = lrn_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.values = []
        self.classes = classes
        self.trained = False
        self.reset = reset
        self.train_set = None
        self.test_set = None
        self.extractor = Extractor()
        
        if os.path.exists("wsnetwork.xml") is False:
            self.ann = buildNetwork(self.in_nodes, self.hid_layers, self.out_nodes, bias=True, \
                recurrent=False, outclass=SoftmaxLayer)
        else:
            self.ann = NetworkReader.readFrom("wsnetwork.xml")
            self.trained = True
            
    def addSetFromDir(self, directory, testing=False, testWhileTrain=False, sound_class=0):
        """
        Generate a data set for the files in the given directory. The data set
        consists of tuples in the following format:
        [feature_vector, audio_class]
        """
        
        #----TRAINING----
        if not testing:
            self.train_set = self.extractor.getFeatureSet(directory, mfcc=True, \
                key_strength=True, spectral_flux=True, zerocrossingrate=True)
            return self.train_set
        
        #----TESTING----
        else:
            self.test_set = self.extractor.getFeatureSet(directory, mfcc=True, \
                key_strength=True, spectral_flux=True, zerocrossingrate=True)
            return self.test_set
            
    def train(self, epochs=50):
        """
        Train the network for a specified number of epochs, or alternatively
        train until the network converges.
        """
        self.train_set._convertToOneOfMany()
        trainer = BackpropTrainer(self.ann, learningrate=self.lrn_rate, dataset=self.train_set, \
            momentum=self.momentum, verbose=True, weightdecay=self.weight_decay)
            
        print "\n*****Starting training..."
        for i in range(50):
            trainer.trainEpochs(50)
            # trainer.trainUntilConvergence()
            self.report_error(trainer)
        print "\n*****Training finished!*****"
        NetworkWriter.writeToFile(self.ann, "wsnetwork.xml")
        print "Network successfully saved to wsnetwork.xml"
        
    def report_error(self, trainer):
        trnresult = percentError(trainer.testOnClassData(), self.train_set['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=self.test_set), self.test_set['class'])
        print "\n------------> epoch: %4d" % trainer.totalepochs, "  + \
            train error: %5.2f%%" % trnresult + "    test error:" + "%5.2f%%" % tstresult
    
    def testOnDir(self, directory, audio_class=0):
        """Test the features for an audio classifier.
        Set up the test classification set, and populate it with samples.
        Currently samples are the same as the training samples. 07/11/15
        """
        
        print "\n*****Testing on audio files in " + directory + "...\n"
        #Generate the testing data set
        tstdata = self.test_set
        # print "testdata: \n" + str(tstdata)
        for i in range(tstdata.getLength()):
            one_sample = ClassificationDataSet(self.in_nodes, nb_classes=self.out_nodes, \
                class_labels=self.classes)
            target_class = (tstdata.getSample(index=i)[1]).argmax()
            one_sample.addSample(tstdata.getSample(index=i)[0], \
                target_class)
            # print "SAMPLE CONTENT: " + str(tstdata.getSample(index=i))
            out = self.ann.activateOnDataset(one_sample)
            print "\nActivation values: " + str(out)
            out = out.argmax()  # the highest output activation gives the class
            print "TARGET: " + str(target_class) + "  |  OUTPUT CLASS: " +   str(out) \
                + " -> " + str(self.classes[out])
        
        print "Error: " + str(percentError(out, tstdata['class'])) + "%" + "\n**"
        # tstdata.clear()
        #Invoke the actiovation function to classify the test data.
    
    def print_info(self):
        print "Audio classes: MUSIC, VOICE, AMBIENT, SILENCE"
        print ("                      PARAMETERS\n"
             + "----> Input neurons: " + str(self.in_nodes) + "\n"
             + "----> Output neurons: " + str(self.out_nodes) + "\n"
             + "----> Number of hidden layers: " + str(self.hid_layers) + "\n"
             + "----> Learning rate: " + str(self.lrn_rate) + "\n"
             + "----> Momentum: " + str(self.momentum) + "\n"
             + "----> Weight Decay: " + str(self.weight_decay) + "\n"
             + "__________________________________________________________"
               )
    

#-------------------------------MAIN PROGRAM-----------------------------------

if __name__ == "__main__":

    print (
         ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n"
        +"                         [whatsound]\n"
        +"                           *******\n"
        +"              Neural network audio classification\n"
        +". . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n"
        )



    # Initialise network
    network = NeuralNetwork()
    network.print_info()
    train = True
    if network.trained:
        prompt = "Network is already trained. Re-run training? (y/n) "
        inc_prompt = "Invalid input. Re-run training? (y/n) "
        # Check if user wants to re-train network
        train_toggle = raw_input(prompt) \
            if sys.version_info[0] < 3 \
            else input(prompt)
        while train_toggle not in 'YyNn':
            train_toggle = raw_input(inc_prompt) \
                if sys.version_info[0] < 3 \
                else input(inc_prompt)
        if train_toggle in 'Nn':
            train = False

    if train:
        network = NeuralNetwork(reset=True)
        print "Neural network reset. Starting..."
    # # Add training data
    # print "\n *Extracting features..."
    #
    # # ------------ TRAINING DATA SET -----------------------
    #
    # network.addSetFromDir("../samples/train/")

    # ------------------ TESTING DATA SET --------------------------

    network.addSetFromDir("../samples/test/", testing=True)


    if train:
        # Start training
        network.train()

    # Test on directories
    network.testOnDir("../samples/test/")
