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
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import normalize
import numpy as np
from numpy import array
import WhatSound_global_data
from WhatSound_extractor import meanMfcc
import yaml
import essentia
from __builtin__ import file
import os


class NeuralNetwork():
    
    def __init__(self,
                    in_nodes=WhatSound_global_data.N_input,
                    weight_decay=WhatSound_global_data.Weight_decay,
                    hid_layers=WhatSound_global_data.N_hidden_layers, 
                    out_nodes=WhatSound_global_data.N_output, 
                    lrn_rate=WhatSound_global_data.Learning_rate, 
                    momentum=WhatSound_global_data.Momentum):
        
        # INITIALISE NETWORK PARAMS
        self.in_nodes = in_nodes
        self.hid_layers = hid_layers
        self.out_nodes = out_nodes
        self.lrn_rate = lrn_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.values = []
        self.classes = ["music", "voice", "ambient"]
        
        self.data_set = ClassificationDataSet(13, nb_classes=self.out_nodes
            , class_labels=['music', 'voice', 'ambient'])
            
        self.ann = buildNetwork(13, self.hid_layers, self.out_nodes, bias=True, \
            recurrent=False, outclass=SigmoidLayer)
    
    """
    Generate a data set for the files in the given directory. The data set 
    consists of tuples in the following format:
    [feature_vector, audio_class]
    """
    def addSetFromDir(self, directory, testing=False, sound_class=0):
        #Normalizer from scikit-learn
        norm_scaler = MaxAbsScaler()
        
        files = os.listdir(directory)
        samples = [""] * len(files)
        for i in range(len(files)):
            samples[i] = directory + files[i]
            
        #----TESTING----
        if (testing == True):            
            # Add the features to the data set
            tstdata = ClassificationDataSet(13, nb_classes=self.out_nodes, \
                class_labels=['music', 'voice', 'ambient'])
                
            for i in range(len(samples)):
                feature_vector = meanMfcc(samples[i])
                # print "Feature vector: " + str(feature_vector)
                feature_vector_norm = norm_scaler.fit_transform(feature_vector)
                print "Normalized: " + str(feature_vector_norm)
                tstdata.addSample(feature_vector_norm, sound_class) 
            tstdata._convertToOneOfMany()
            return tstdata
            
        #----TRAINING----
        # Add the features to the data set
        for i in range(len(samples)):
            feature_vector = meanMfcc(samples[i])
            # print "Feature vector: " + str(feature_vector)
            feature_vector_norm = norm_scaler.fit_transform(feature_vector)
            print "Normalized feature vector: " + str(feature_vector_norm)
            self.data_set.addSample(feature_vector_norm, sound_class)
        return self.data_set
    
    """
    Train the network for a specified number of epochs, or alternatively 
    train until the network converges.
    """
    def train(self, epochs=50):
        
        self.data_set._convertToOneOfMany()
        
        trainer = BackpropTrainer(self.ann, learningrate=self.lrn_rate, dataset=self.data_set, \
            momentum=self.momentum, verbose=False, weightdecay=self.weight_decay)
        
        print "\n*****Starting training..."
        for i in range(15):
            trainer.trainEpochs(200)
            # trainer.trainUntilConvergence()
            trnresult = percentError(trainer.testOnClassData(dataset=self.data_set), self.data_set['class'])
            self.report_error(trainer, self.data_set)
        print "\n*****Training finished!*****"
            
            
    def report_error(self, trainer, trndata):
        trnresult = percentError(trainer.testOnClassData(), trndata['class'])
        print "------------> epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult
        
        
    def testOnDir(self, directory, audio_class=0):
        """Test the features for an audio classifier. 
        Set up the test classification set, and populate it with samplesq.
        Currently samples are the same as the training samples. 07/11/15
        """ 
        
        print "\n*****Testing on audio files in " + directory + "...\n"
        #Generate the testing data set
        tstdata = self.addSetFromDir(directory, testing=True, sound_class=audio_class)
        print "testdata: \n" + str(tstdata)
        for i in range(tstdata.getLength()):            
            one_sample = ClassificationDataSet(13, nb_classes=self.out_nodes, \
                class_labels=['music', 'voice', 'ambient'])
            one_sample.addSample(tstdata.getSample(index=i)[0], audio_class)    
            out = self.ann.activateOnDataset(one_sample)
            print "\n     Output activation -->: " + str(out)
            out = out.argmax()  # the highest output activation gives the class
            print ("     ===================================================" +\
                   "\n     #   Output: The test tuple is of type: **" 
                    + str(self.classes[out]) + "**" + "   #" +\
                   "\n     ===================================================")
                            
                
        print "Error: " + str(percentError(out, tstdata['class'])) + "%"
        tstdata.clear()
        #Invoke the actiovation function to classify the test data. 
    
    def printParams(self):
        print ("===================== PARAMETERS ==========================\n"
             + "----> Input neurons: " + str(self.in_nodes) + "\n"
             + "----> Output neurons: " + str(self.out_nodes) + "\n"
             + "----> Number of hidden layers: " + str(self.hid_layers) + "\n"
             + "----> Learning rate: " + str(self.lrn_rate) + "\n"
             + "----> Momentum: " + str(self.momentum) + "\n"
             + "----> Weight Decay: " + str(self.weight_decay) + "\n"
             + "==========================================================="
               )


#-------------------------------MAIN PROGRAM-----------------------------------

if __name__ == "__main__":
    
    print (
        "------------------------------------------------------------\n"
        +"=========================[whatsound]=======================\n"
        +"                           *******                         \n"
        +"              Neural network audio classification          \n"
        +"===========================================================\n"
        )
    
    # Initialise network
    network = NeuralNetwork()
    network.printParams()
    # Add training data
    print "\n *Extracting features..."
    network.addSetFromDir("../samples/train/guitar/", sound_class=0)
    network.addSetFromDir("../samples/train/male voice/", sound_class=1)
    # Start training
    network.train()
    
    # Test on directories
    network.testOnDir("../samples/test/music/", audio_class=0)
    network.testOnDir("../samples/test/voice/", audio_class=1)
