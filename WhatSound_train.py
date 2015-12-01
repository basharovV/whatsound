"""
    'whatsound' - A neural network approach for classifying audio
    Author: Vyacheslav Basharov
    Version
"""
from WhatSound_extractor import *
from WhatSound_neural import *
from numpy import array
import yaml
import essentia
from __builtin__ import file

def printParams():
    print (
           "Input neurons: " + str(nn_global_data.N_input) + "\n"
           + "Output neurons: " + str(nn_global_data.N_output) + "\n"
           + "Number of hidden layers: " + str(nn_global_data.N_hidden_layers) + "\n"
           + "Learning rate: " + str(nn_global_data.Learning_rate) + "\n"
           )

def testOnAllData():
    """Test the features for an audio classifier. 
    Set up the test classification set, and populate it with samples.
    Currently samples are the same as the training samples. 07/11/15
    """ 
    #Add the samples to the testing data set
    tstdata = ClassificationDataSet(N_input, nb_classes=N_output, class_labels=['music', 'voice', 'ambient'])
    
    print "TESTING ROUND..."
    for i in range(len(samples)):
        tstdata.addSample(values[i], samples[i][1])
        tstdata._convertToOneOfMany()
        print "\nData set N. : ", i
        print "Data set values: " , values[i]
        out = ann.activateOnDataset(tstdata)
        out = out.argmax()  # the highest output activation gives the class
        print "OUTPUT: " , out
        print ("Output: The test file " + str(samples[i][0])  + " is of type: *****" 
            + str(classes[out]) + "*****")
        tstdata.clear()
    #Invoke the actiovation function to classify the test data. 
           

# net = buildNetwork(2, 3, 1)
#
# net.activate([2, 1])
# print net['in']

N_input = nn_global_data.N_input;
N_hidden = nn_global_data.N_hidden_layers;
N_output = nn_global_data.N_output;


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
testFile = samples[[0][0]]

print "Calculating mfcc values..."
            
# Add the features to the input vectors
for i in range(len(samples)):
    values.append(meanMfcc(samples[i][0]))

print "...done! \n\nStarting training"


# Create the dataset with specific params
DS = ClassificationDataSet(N_input, nb_classes=N_output, class_labels=['music', 'voice', 'ambient'])

# Populate the dataset with the feature values
for i in range(len(values)):
    DS.addSample(values[i], samples[i][1])

data = DS
data.setField('class', [[0], [1], [0], [2], [2]])

# 
# trndata = ClassificationDataSet(13, nb_classes=3, class_labels=['music', 'voice', 'ambient'])
# for n in xrange(0, trndata_temp.getLength()):
#     trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )
    
data._convertToOneOfMany()

ann = buildNetwork(N_input, N_hidden, N_output, bias=True, recurrent=False, outclass=SoftmaxLayer)
trainer = BackpropTrainer( ann, learningrate=0.5, dataset=data, momentum=0.9, verbose=True, weightdecay=0.01)


trainer.trainUntilConvergence(dataset=data)
trnresult = percentError( trainer.testOnClassData(), data['class'])
        
print ("epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult)

testOnAllData()

if __name__ == "__main__":
    mainTrain()
