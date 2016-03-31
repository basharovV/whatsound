"""
    'WhatSound' - A neural network approach for classifying audio
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
from WS_utils import *
import yaml
import essentia
from __builtin__ import file
import os
import sys
import argparse


class NeuralNetwork():
    
    def __init__(self,
                    weights_file=WS_global_data.weights_path,
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
        self.test_set_freqs = None
        self.split = split if WS_global_data.split_enabled else None
        self.extractor = Extractor()
        self.weights_file = weights_file
        
        # Outclass determines the activation function at the output layer
        # This can be
        #- Softmax function - supposedly better for classicication
        #- Sigmoid function -
        
        if os.path.exists(self.weights_file) is False or reset:
            self.ann = buildNetwork(self.in_nodes, self.hid_nodes, self.out_nodes, bias=True, \
                recurrent=False, outclass=SigmoidLayer)
        else:
            # print "loading weights file " + self.weights_file
            self.ann = NetworkReader.readFrom(self.weights_file)
            self.trained = True
            
    def set_weights_file(self, filepath):
        self.weights_file = filepath
        
    def reconfigure_network(self, **kwargs):
        # print kwargs
        self.weight_decay =  kwargs.pop('weight_decay')
        self.hid_nodes =  kwargs.pop('hid_nodes')
        self.lrn_rate =  kwargs.pop('lrn_rate')
        self.momentum =  kwargs.pop('momentum')
        self.ann.reset()
        self.ann = buildNetwork(self.in_nodes, self.hid_nodes, self.out_nodes, bias=True, \
            recurrent=False, outclass=SigmoidLayer)
            
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
        data_set = self.extractor.get_dir_dataset(directory)
        if WS_global_data.split_enabled:
            self.train_set, self.test_set = data_set.\
                splitWithProportion(self.split)
            self.train_set = supervised_to_classification(self.train_set)
            self.test_set = supervised_to_classification(self.test_set)
            self.train_set._convertToOneOfMany()
            self.test_set._convertToOneOfMany()
            self.test_set_freqs = get_set_freqs(self.test_set)
        else:
            if not testing:
                self.train_set = data_set
                train_freqs = get_set_freqs(self.train_set)
                # print "Train freqs: " + str(train_freqs)
            else:
                self.test_set = data_set
                self.test_set_freqs = get_set_freqs(self.test_set)
                # print str(self.test_set_freqs)
    
    def train(self, epochs=50, verbose=True):
        """
        Train the network for a specified number of epochs, or alternatively
        train until the network converges.
        """
        
        # self.train_set._convertToOneOfMany()
        trainer = BackpropTrainer(self.ann, learningrate=self.lrn_rate, \
                dataset=self.train_set, momentum=self.momentum, \
                verbose=verbose, weightdecay=self.weight_decay)
        if verbose:
            print "\n*****Starting training..."
        interval = 50
        for i in range(50):
            trainer.trainEpochs((int) (epochs / 50))
            # trainer.trainUntilConvergence()
            if verbose:
                self.report_error(trainer)
        if verbose:
            print "\n*****Training finished!*****"
        NetworkWriter.writeToFile(self.ann, self.weights_file)
        if verbose:
            print "Network successfully saved to " + self.weights_file
            
    def report_error(self, trainer):
        # print self.train_set
        trnresult = percentError(trainer.testOnClassData(), self.train_set['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=self.test_set), self.test_set['class'])
        print "\n------------> epoch: %4d" % trainer.totalepochs, "  + \
            train error: %5.2f%%" % trnresult + "    test error:" + "%5.2f%%" % tstresult
    
    def test(self, confusion_matrix=False):
        tstdata = self.test_set
        sample_count = self.test_set.getLength()
        print "Test data of size: " + str(sample_count)
        correct_count = 0
        
        # Matrix to indicate the number of correct and incorrect predictions
        hitrate_matrix = np.matrix([[0 for i in range(WS_global_data.N_output)] \
                            for i in range(WS_global_data.N_output)])
        # tstdata._convertToOneOfMany()
        # print "testdata: \n" + str(tstdata)
        for i in range(len(tstdata)):
            one_sample = ClassificationDataSet(self.in_nodes, nb_classes=self.out_nodes, \
                class_labels=self.classes)
            target_class = np.argmax(tstdata.getSample(index=i)[1])
            one_sample.addSample(tstdata.getSample(index=i)[0], target_class)
            one_sample._convertToOneOfMany()
            # print "ONE SAMPLE: " + str(one_sample)
            activation = self.ann.activateOnDataset(one_sample)
            out = activation.argmax()
            if out == target_class:
                correct_count+=1
            hitrate_matrix[target_class, out] = hitrate_matrix[target_class, out] + 1
        
        print hitrate_matrix
        if confusion_matrix:
            test_info = "Score : %s / %s" % (correct_count, sample_count)
            test_accuracy = correct_count / float(sample_count)
            print test_info, test_accuracy
            return get_confusion_matrix(hitrate_matrix, self.test_set_freqs)
        
        test_info = "Score : %s / %s" % (correct_count, sample_count)
        test_accuracy = correct_count / float(sample_count)
        return (test_accuracy, test_info)
    
    def test_on_file(self, filepath, audio_class=None, verbose=False):
        # print "\n*****Testing on audio files in " + directory + "...\n"
        # Generate the testing data set in 3D list form
        tstdata = self.extractor.get_single_dataset(audio_class=audio_class, filepath=filepath, as_array=True)
        one_sample = ClassificationDataSet(self.in_nodes, nb_classes=self.out_nodes, \
            class_labels=self.classes)
        target_class = audio_class if audio_class != None else 0
        one_sample.addSample(tstdata[2][0], \
            target_class)
        one_sample._convertToOneOfMany()
        # print "ONE SAMPLE: " + str(one_sample)
        activation = self.ann.activateOnDataset(one_sample)
        out = activation.argmax()
        if verbose:
            print "______________________________________________________" + \
                    "\n\n File: " + tstdata[0][0] + \
                    "\n Activation values: " + str(activation) + \
                    "\n Target : " + str(self.classes[target_class]) + "  |  Output: " + \
                    str(self.classes[out])
        return self.classes[out]
    
    def test_on_signal(self, data, audio_class=None):
        """
        Test the trained network on a raw audio signal in a numpy array format.

        args:
            data: The numpy array containing the audio frames
        kwargs:
            audio_class (int): The index for the target class corresponding to
                the signal. By default no class is provided, as in most cases
                the data will be recorded directly from an input device.
        """
        tstdata = self.extractor.get_single_dataset(audio_class=0, as_array=True, signal=data)
        one_sample = ClassificationDataSet(self.in_nodes, nb_classes=self.out_nodes, \
            class_labels=self.classes)
        target_class = audio_class if audio_class != None else 0
        print "TSTDATA: " + str(tstdata[2][0])
        activation = self.ann.activate(tstdata[2][0])
        out = activation.argmax()
        print "______________________________________________________" + \
                "\n\nActivation values: " + str(activation) + \
                "\nOutput: " + str(self.classes[out]) + "\n"
    
    def test_on_dir(self, directory, audio_class=None, structured=True, verbose=True):
        """
        Test the trained network on a directory containing audio files.

        args:
            directory (string): Relative path of the directory.
        kwargs:
            structured (bool): Flag for specifying if the directory follows the
                audio class structure i.e contains 4 correctly named
                sub-directories for each of the audio classes.
        """
        # Generate the testing data set in 3D list form
        tstdata = self.extractor.get_dir_dataset(directory, audio_class=audio_class,
                                                 as_array=True,
                                                 structured=structured)
        sample_count = len(tstdata[0])
        correct_count = 0
        # tstdata._convertToOneOfMany()
        # print "testdata: \n" + str(tstdata)
        for i in range(len(tstdata[0])):
            one_sample = ClassificationDataSet(self.in_nodes, nb_classes=self.out_nodes, \
                class_labels=self.classes)
            
            target_class = int(tstdata[1][i]) if audio_class else 0
            
            one_sample.addSample(tstdata[2][i], \
                target_class)
            one_sample._convertToOneOfMany()
            # print "ONE SAMPLE: " + str(one_sample)
            activation = self.ann.activateOnDataset(one_sample)
            out = activation.argmax()
            if audio_class:
                if out == target_class:
                    correct_count+=1
            print "______________________________________________________" + \
                    "\n\n File: " + tstdata[0][i]
            if verbose:
                print "\n Activation values: " + str(activation)
            if audio_class:
                print "Target : " + str(self.classes[target_class])
            print "|---> Output: " + str(self.classes[out])
        if audio_class:
            print "Score : %s / %s" % (correct_count, sample_count)
        else:
            print "\nProcessed files: %s" % sample_count
    
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

# -------------------------------MAIN PROGRAM-----------------------------------

if __name__ == "__main__":

    print (
         ". . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n"
        +"                         [WS]\n"
        +"                           *******\n"
        +"              Neural network audio classification\n"
        +". . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\n"
        )
        
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", help="The path of the weights file")
    parser.add_argument("-d", "--dataset", help="The path of the features dataset")
    parser.add_argument("-s", "--split", help="The proportion to split the dataset")
    args = parser.parse_args()
    
    weights_file = args.weights
    dataset_path = args.dataset
    split_proportion = float(args.split)
    
    # Initialise network
    if weights_file != None:
        network = NeuralNetwork(weights_file=weights_file, split=split_proportion)
    else:
        network = NeuralNetwork(split=split_proportion)
    test_params = {
                'hid_nodes': 8,
                'lrn_rate': 0.01,
                'weight_decay': 1e-05,
                'momentum': 0.8
                }
                
    # network.reconfigure_network(**test_params)
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
    if network.split:
        network.add_set_from_dir(WS_global_data.data_path)
    # Otherwise add training and testing set separately
    else:
        network.add_set_from_dir(WS_global_data.train_path)
        network.add_set_from_dir (WS_global_data.test_path, testing=True)
    
    # print "PRINTING TRAINING DATA to FILE"
    # network.exportTrainingData()
    
    if train:
        # Start training
        network.train(epochs=3000)
    
    # Test on the testing directory
    if split_proportion != None:
        print str(network.test(confusion_matrix=True))
    else:
        network.test_on_dir(WS_global_data.test_path)
