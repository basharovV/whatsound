"""
    'WS' - A neural network approach for classifying audio
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
from pybrain.structure           import FeedForwardNetwork
from pybrain.structure.modules   import SoftmaxLayer, SigmoidLayer, TanhLayer
from pybrain.tools.customxml import NetworkReader, NetworkWriter
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.preprocessing import normalize
import numpy as np
from numpy import array
import WS_global_data
from WS_extractor import *
import yaml
import essentia
from __builtin__ import file
import os
import sys


class NeuralNetwork():
    
    def __init__(self,
                    in_nodes=WS_global_data.N_input,
                    weight_decay=WS_global_data.Weight_decay,
                    hid_nodes=WS_global_data.N_hidden_nodes,
                    out_nodes=WS_global_data.N_output,
                    lrn_rate=WS_global_data.Learning_rate,
                    momentum=WS_global_data.Momentum,
                    classes=WS_global_data.Classes,
                    split=WS_global_data.split_proportion,
                    reset=False):
        
        # INITIALISE NETWORK PARAMS
        self.in_nodes = in_nodes
        self.hid_nodes = hid_nodes
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
        self.split = split if WS_global_data.split_enabled else 0.0
        self.extractor = Extractor()
        
        """
        Outclass determines the activation function at the output layer
        This can be
        - Softmax function - supposedly better for classicication
        - Sigmoid function - 
        """
        
        
        if os.path.exists("wsnetwork.xml") is False or reset:
            self.ann = buildNetwork(self.in_nodes, self.hid_nodes, self.out_nodes, bias=True, \
                recurrent=True, outclass=SigmoidLayer)
        else:
            self.ann = NetworkReader.readFrom("wsnetwork.xml")
            self.trained = True
            
    def makeNetwork(self):
         self.ann = FeedForwardNetwork()
         inLayer = LinearLayer(self.in_nodes)
         hiddenLayer = SigmoidLayer(self.hid_nodes)
         outLayer = LinearLayer(self.out_nodes)
         
         
    def add_set_from_dir(self, directory, testing=False):
        """
        Generate a data set for the files in the given directory. 
        
        args:
            directory (String): The path containing the audio files in their
                                associated directory.
        kwargs:
            testing: Flag to set what the data will be used for, training or
                     testing.
                
        """
        data_set = self.extractor.getFeatureSet(directory)
        if WS_global_data.split_enabled:
            self.train_set, self.test_set = data_set.\
                splitWithProportion(WS_global_data.split_proportion)
        else:
            if not testing:
                self.train_set = data_set
            else:
                self.test_set = data_set
            
    def train(self, epochs=50):
        """
        Train the network for a specified number of epochs, or alternatively
        train until the network converges.
        """
        # self.train_set._convertToOneOfMany()
        trainer = BackpropTrainer(self.ann, learningrate=self.lrn_rate, \
                dataset=self.train_set, momentum=self.momentum, \
                verbose=True, weightdecay=self.weight_decay)
        print "\n*****Starting training..."
        for i in range(25):
            trainer.trainEpochs(100)
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
    
    def test_on_file(self, filepath, audio_class):
        # print "\n*****Testing on audio files in " + directory + "...\n"
        #Generate the testing data set in 3D list form
        tstdata = self.extractor.get_file_dataset(filepath, audio_class, as_array=True)
        one_sample = ClassificationDataSet(self.in_nodes, nb_classes=self.out_nodes, \
            class_labels=self.classes)
        target_class = audio_class
        one_sample.addSample(tstdata[2][0], \
            target_class)
        one_sample._convertToOneOfMany()
        # print "ONE SAMPLE: " + str(one_sample)
        activation = self.ann.activateOnDataset(one_sample)
        out = activation.argmax()
        print "______________________________________________________" + \
                "\n\n File: " + tstdata[0][0] + \
                "\n Activation values: " + str(activation) + \
                "\n Target : " + str(self.classes[target_class]) + "  |  Output: " + \
                str(self.classes[out])
    
    def test_on_dir(self, directory):
        """Test the features for an audio classifier.
        Set up the test classification set, and populate it with samples.
        Currently samples are the same as the training samples. 07/11/15
        """
        
        # print "\n*****Testing on audio files in " + directory + "...\n"
        #Generate the testing data set in 3D list form
        tstdata = self.extractor.getFeatureSet(directory, as_array=True)
        # tstdata._convertToOneOfMany()
        # print "testdata: \n" + str(tstdata)
        for i in range(len(tstdata[0])):
            one_sample = ClassificationDataSet(self.in_nodes, nb_classes=self.out_nodes, \
                class_labels=self.classes)
            target_class = tstdata[1][i]
            one_sample.addSample(tstdata[2][i], \
                target_class)
            one_sample._convertToOneOfMany()
            # print "ONE SAMPLE: " + str(one_sample)
            activation = self.ann.activateOnDataset(one_sample)
            out = activation.argmax()
            print "______________________________________________________" + \
                    "\n\n File: " + tstdata[0][i] + \
                    "\n Activation values: " + str(activation) + \
                    "\n Target : " + str(self.classes[target_class]) + "  |  Output: " + \
                    str(self.classes[out])
            # print "SAMPLE CONTENT: " + str(tstdata.getSample(index=i))
              # the highest output activation gives the class
            # print "TARGET: "         
            # print "Error: " + str(percentError(out, target_class)) + "%" + "\n**"
        # tstdata.clear()
        #Invoke th
    def print_info(self):
        print "Audio classes: MUSIC, VOICE, AMBIENT, SILENCE"
        print ("                      PARAMETERS\n"
             + "----> Input neurons: " + str(self.in_nodes) + "\n"
             + "----> Output neurons: " + str(self.out_nodes) + "\n"
             + "----> Number of hidden layers: " + str(self.hid_nodes) + "\n"
             + "----> Learning rate: " + str(self.lrn_rate) + "\n"
             + "----> Momentum: " + str(self.momentum) + "\n"
             + "----> Weight Decay: " + str(self.weight_decay) + "\n"
             + "__________________________________________________________"
               )
    
    def exportTrainingData(self):
        export_file = open("traindata.txt", "w+")
        for i in range(self.train_set.getLength()):
            export_file.write(str(self.train_set.getSample(index=i)) + "\n")
        export_file.close()

#-------------------------------MAIN PROGRAM-----------------------------------

if __name__ == "__main__":

    print (
         ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n"
        +"                         [WS]\n"
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
    print "\n *Extracting features..."
    
    # If split is enabled, only add one data set, which will be split into two.
    if (network.split):
        network.add_set_from_dir(WS_global_data.data_path)
    # Otherwise add training and testing set separately
    else:
        network.add_set_from_dir(WS_global_data.train_path)
        network.add_set_from_dir (WS_global_data.test_path, testing=True)

    print "PRINTING TRAINING DATA to FILE"
    network.exportTrainingData()
    
    if train:
        # Start training
        network.train()

    # Test on the testing directory
    network.test_on_dir(WS_global_data.test_path)
