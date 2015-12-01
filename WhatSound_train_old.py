"""
    'whatsound' - A neural network approach for classifying audio
    Author: Vyacheslav Basharov
    Version
"""

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
from numpy import array
from WhatSound_global_data import *
from WhatSound_extractor import meanMfcc
import yaml
import essentia
from __builtin__ import file
import os

class NeuralNetwork():
    
    def __init__(self, samples, hid_nodes=10, 
        hid_layers=1, out_nodes=3, lrn_rate=0.05, momentum=0.1):
        
        # INITIALISE NETWORK PARAMS
        self.samples = samples
        self.hid_nodes = hid_nodes
        self.hid_layers = hid_layers
        self.out_nodes = out_nodes
        self.lrn_rate = lrn_rate
        self.momentum = momentum
        self.values = []
        self.classes = ["music", "voice", "ambient"]
    
    def train(self, epochs=50):
        # EXTRACT THE AUDIO DESCRIPTORS FOR THE GIVEN FILES
        
        in_nodes = 0
        # Add the features to the input vectors
        for i in range(len(self.samples)):
            feature_vector = meanMfcc(self.samples[i][0])
            self.values.append(feature_vector)
            if in_nodes == 0:
                in_nodes = len(feature_vector)
                
        # Create the dataset with specific params
        DS = ClassificationDataSet(13, nb_classes=self.out_nodes
            , class_labels=['music', 'voice', 'ambient'])
        
        # Populate the dataset with the feature values
        for i in range(len(self.values)):
            DS.addSample(self.values[i], self.samples[i][1])
        
        data = DS
        data.setField('class', [[0], [1], [0], [2], [2]])
        
        # 
        # trndata = ClassificationDataSet(13, nb_classes=3, class_labels=['music', 'voice', 'ambient'])
        # for n in xrange(0, trndata_temp.getLength()):
        #     trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
        
        data._convertToOneOfMany()
        self.ann = buildNetwork(13, self.hid_layers, self.out_nodes, bias=True, recurrent=False, outclass=SoftmaxLayer)
        trainer = BackpropTrainer(self.ann, learningrate=self.lrn_rate, dataset=data, momentum=self.momentum, verbose=True, weightdecay=0.1)
        tstdata = self.make_test_sample()
        
        for i in range(20):
            trainer.trainEpochs(5)
            # trainer.trainUntilConvergence()
            trnresult = percentError(trainer.testOnClassData(dataset=data), data['class'])
            self.report_error(trainer, data, tstdata)
            
            
    def report_error(self, trainer, trndata, tstdata):
        trnresult = percentError(trainer.testOnClassData(), trndata['class'])
        tstresult = percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
        print "epoch: %4d" % trainer.totalepochs, "  train error: %5.2f%%" % trnresult, "  test error: %5.2f%%" % tstresult
    
    def make_test_sample(self, sample=1):
        """Test the features for an audio classifier. 
        Set up the test classification set, and populate it with samplesq.
        Currently samples are the same as the training samples. 07/11/15
        """ 
        #Add the samples to the testing data set
        tstdata = ClassificationDataSet(13, nb_classes=self.out_nodes, class_labels=['music', 'voice', 'ambient'])
        tstdata.addSample(self.values[sample], self.samples[sample][1])
        tstdata._convertToOneOfMany()
        # out = self.ann.activateOnDataset(tstdata)
        # out = out.argmax()  # the highest output activation gives the class
        # tstdata.clear()
        return tstdata        
        
    def testOnSample(self, sample):
        """Test the features for an audio classifier. 
        Set up the test classification set, and populate it with samplesq.
        Currently samples are the same as the training samples. 07/11/15
        """ 
        
        #Add the samples to the testing data set
        tstdata = ClassificationDataSet(13, nb_classes=self.out_nodes, class_labels=['music', 'voice', 'ambient'])
        
        
        tstdata.addSample(sample[0][0], sample[0][1])
        tstdata._convertToOneOfMany()
        print "Data set values: " , str(sample[0][0])
        out = self.ann.activateOnDataset(tstdata)
        out = out.argmax()  # the highest output activation gives the class
        print "OUTPUT: " , out
        print ("Output: The test file " + str(sample[0][2])  + " is of type: *****" 
            + str(self.classes[out]) + "*****")
        print "PERCENT ERROR: " + str(percentError(out, tstdata['class']))
        tstdata.clear()
        #Invoke the actiovation function to classify the test data. 
    

def printParams():
    print (
           "Input neurons: " + str(N_input) + "\n"
           + "Output neurons: " + str(N_output) + "\n"
           + "Number of hidden layers: " + str(N_hidden_layers) + "\n"
           + "Learning rate: " + str(Learning_rate) + "\n"
           )


def start():
    # net = buildNetwork(2, 3, 1)
    #
    # net.activate([2, 1])
    # print net['in']
    
    print (
         "_____________________________[whatsound]____________________________\n"
        +"                               *******                              \n"
        +"                  Neural network audio classification               \n"
        +"--------------------------------------------------------------------\n"
        )
    
    printParams()
    values = []
    # 
    # with open('mean_mfcc.sig', 'r') as file:
    #     features = yaml.load(file)
    # 
    # input = features["mean_mfcc"][0]
    # print input
    
    classes = ["music", "voice", "ambient"]
    
    # samples = [ ['samples/music-strings1.wav', 0], 
    #             ['samples/voice1.mp3', 1],
    #             ['samples/music-2.wav', 0],
    #             ['samples/ambient-1.wav', 2],
    #             ['samples/ambient-2.wav', 2]
    #             ]
                
    # Train on all guitar samples    
    path_guitar = "sounds/guitar/"
    guitarFiles = os.listdir(path_guitar)
    samples = [[0 for j in range(2)] for i in range(15)]
    for i in range(len(guitarFiles)):
        samples[i][0] = path_guitar + guitarFiles[i]
        samples[i][1] = 0 #for music
        
    
    # ----------------- FILE TO TEST ON THE TRAINING DATA ---------------------
    testFile = samples[[0][0]]
    
    network = NeuralNetwork(samples)
    network.train()
    
    test = [['', 0, '']]
    test[0][0] = meanMfcc("samples/music-3.mp3")
    test[0][1] = 1
    test[0][2] = "samples/music-3.mp3"
    network.testOnSample(test)

if __name__ == "__main__":
    start()
