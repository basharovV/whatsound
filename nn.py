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
    
features = []

with open('mean_mfcc.sig', 'r') as file:
    features = yaml.load(file)

input = features["mean_mfcc"][0]
# print input
        
DS = ClassificationDataSet(13, nb_classes=2, class_labels=['music', 'speech'])
# DS.addSample(input, 1)
data = DS
data._convertToOneOfMany()
ann = buildNetwork(data.indim, 5, data.outdim, bias=True, recurrent=False, outclass=SoftmaxLayer)
trainer = BackpropTrainer( ann, dataset=data, momentum=0.1, verbose=True, weightdecay=0.01)


for i in range(20):
    trainer.trainEpochs(1)
    trnresult = percentError( trainer.testOnClassData(),
                                  data['class'] )
    print ("epoch: %4d" % trainer.totalepochs, \
              "  train error: %5.2f%%" % trnresult)
    out = ann.activateOnDataset(data)
    out = out.argmax(axis=1)  # the highest output activation gives the class
    print "Output: " + str(out)
