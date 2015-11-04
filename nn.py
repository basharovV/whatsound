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
import nn_global_data
import yaml
import essentia
from __builtin__ import file
from  features import meanMfcc

def printParams():
    print (
           "Input neurons: " + str(nn_global_data.N_input) + "\n"
           + "Output neurons: " + str(nn_global_data.N_output) + "\n"
           + "Number of hidden layers: " + str(nn_global_data.N_hidden_layers) + "\n"
           + "Learning rate: " + str(nn_global_data.Learning_rate) + "\n"
           )

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

samples = [ ['samples/music-strings1.wav', 0], 
            ['samples/voice1.mp3', 1],
            ['samples/music-2.wav', 0],
            ['samples/ambient-1.wav', 2],
            ['samples/ambient-2.wav', 2]
            ]
            
# ----------------- FILE TO TEST ON THE TRAINING DATA ---------------------
testFile = samples[[0][0][0]]
            
# Add the features to the input vectors
for i in range(len(samples)):
    values.append(meanMfcc(samples[i][0]))

# Create the dataset with specific params
DS = ClassificationDataSet(13, nb_classes=3, class_labels=['music', 'voice', 'ambient'])

# Populate the dataset with the feature values
for i in range(len(values)):
    DS.addSample(values[i], samples[i][1])

data = DS
data.setField('class', [[0], [1], [0], [2], [2]])

# 
# trndata = ClassificationDataSet(13, nb_classes=3, class_labels=['music', 'voice', 'ambient'])
# for n in xrange(0, trndata_temp.getLength()):
#     trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
    
tstdata = ClassificationDataSet(13, nb_classes=3, class_labels=['music', 'voice', 'ambient'])
tstdata.addSample( data.getSample(0)[0], data.getSample(0)[1] )

data.setField('class', [[0], [1], [0], [2], [2]])

data._convertToOneOfMany()
tstdata._convertToOneOfMany()

ann = buildNetwork(13, 3, 3, bias=True, recurrent=False, outclass=SoftmaxLayer)
trainer = BackpropTrainer( ann, dataset=data, momentum=0.1, verbose=True, weightdecay=0.01)


for i in range(20):
    trainer.trainEpochs(100)
    trnresult = percentError( trainer.testOnClassData(), data['class'])
    
    print ("epoch: %4d" % trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult)
              
out = ann.activateOnDataset(tstdata)
out = out.argmax()  # the highest output activation gives the class
print ("Output: The test file " + testFile  + " is of type: *****" 
    + str(classes[out]) + "*****")
