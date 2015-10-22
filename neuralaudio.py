###########################################################
#
# CSC D84 - A5 - Neural Networks for classification
#
# Your tas here is to implement the equations and algorithms
# required to train a Neural Network to perform classification
# of hand-written digits. The goal is to strengthen your
# understanding of
#
# Network structure
# Feed-forward computations
# Error computation
# Error back-propagation
#
# Your code will be called by NeuralNets_core_GL
# during training and testing.
#
# You have access to the following global data (all
# prefixed by 'NeuralNets_global_data.' )
#
# N_out		- Number of output neurons (one per
#                 digit in this case, so this is fixed
#		  at 10, digits are 0 to 9)
# N_in		- Number of inputs. Digits in our
#		  problem consist of a matrix of
#		  8x8 pixels, so 64 inputs plus 1
#		  for the bias term = 65
# N_hidden	- Number of hidden-layer units.
#		  we will train networks with
#		  different numbers of units. See
#		  the REPORT.TXT for more details
#		  Note that the actual number of
#		  neurons is N_hidden-1, but we have
#		  one extra entry for the bias term!
# alpha		- The network's learning rate. Set by
#		  the NeuralNets_init() function.
# sig_type	- Type of sigmoidal function to use,
#		  0 -> logistic, 1 -> tanh
#
# W_io		- For networks with NO hidden layer,
#		  this gives the weights connecting
#		  inputs to output neurons.
#		  This array is of size (N_in, N_out)
#		  and W_io(i,j) gives the weight
#		  connecting input i to output j
#
# W_ih		- For networks with hidden layer,
#		  this contains the weights linking
#		  inputs to hidden layer neurons.
#		  the arry has size (N_in, N_hidden-1)
#		  (MIND the -1!)
#
# W_ho		- For networks with hidden layer,
#		  this gives the weights connecting
#		  neurons in the hidden layer to
#		  units in the output layer. The
#		  array is of size (N_hidden, N_out)
#
# Do not add any additional global data.
#
# Starter code: F.J.E. Mar. 2013, Updated Jan. 2014
##########################################################

import NeuralNets_global_data
import random
import math
from numpy import *



def sigmoid(x):
	if NeuralNets_global_data.sig_type:
		return(tanhyp(x))
	else:
		return(logistic(x))

def logistic(x):
##########################################################
#
# TO DO: Implement the Logistic function computation
#
##########################################################
    result = float(1 / (1 + math.exp(-x)))
    return result

def tanhyp(x):
##########################################################
#
# TO DO: Complete the hyperbolic tangent computation
#        using exponentials.
#
##########################################################

    # print "X = " + str(x)
    # print "top" + str(float(math.exp(x) - math.exp(-x)))
    # print "bottom" + str(float(math.exp(x) + math.exp(-x)))
    result = 0
    try:
        result = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    except OverflowError:
        print "OVERFLOW with x: " + str(x)
        if x/x == 1:
            return 1.0
        else:
            return -1.0
    return result

def sigmoid_prime(a):
##########################################################
#
# TO DO: Implement this function to return the derivative
#        of the corresponding sigmoid function in terms
#        of the output of the neuron.
#
##########################################################
    if (NeuralNets_global_data.sig_type):
        ans = (1 - (a **2))
        return ans	# Change this as needed!

    else:
        return a * (1 - a)  # Change this as needed

def FeedForward(input_sample):
##########################################################
# This function computes the feed-forward activation of
# each neuron in the network.
#
# For networks with no hidden layer, the feed-forward pass
# is straightforward. The input sample already contains all
# the information you need (including the bias term).
#
# For networks with a hidden layer, compute the outputs
# for each hidden layer neuron, put them all in a list,
# and add a bias term of 1.0. This list will be the input
# to the output layer.
#
# You should have completed the logistic() or tanhyp()
# function(s) before you can attempt this part.
#
# NOTE: Your code for this part should call the
#                       sigmoid()
#       function. NeuralNets_core_GL will set the sigmoid
#       type, so that either logistic() or tanhyp() will be
#       called depending on what the user specifies.
##########################################################

    outputActivation=zeros(shape=(NeuralNets_global_data.N_out,1))		# Array of activation values for output units
    hiddenActivation=zeros(shape=(NeuralNets_global_data.N_hidden,1))	# Array of activation values for hidden units
    # DO remember that the last entry in hiddenActivation should be a constant
    # bias term = 1.0!

    ##########################################################
    #
    # TO DO: Complete this function to compute the activation
    #        values for each neuron in the network.
    #
    ##########################################################

	#No hidden layer
    # print "NHIDDEN:" + str(NeuralNets_global_data.N_hidden)

    if NeuralNets_global_data.N_hidden == 0:
        for o in range(len(outputActivation)):
            sum_input = 0
            for i in range(len(input_sample)):
                sum_input += NeuralNets_global_data.W_io[i][o] * input_sample[i]
            outputActivation[o] = sigmoid(sum_input)

    #Calculation for hidden layer
    else:
        for h in range(len(hiddenActivation) - 1):
            sum_hidden = 0
            for i in range(len(input_sample)):
                sum_hidden += (NeuralNets_global_data.W_ih[i][h] * input_sample[i])
            hiddenActivation[h] = sigmoid(sum_hidden)
        hiddenActivation[NeuralNets_global_data.N_hidden - 1] = 1.0

        #Feed output from hidden layer into the output neurons

        for o in range(len(outputActivation)):
            sum_input = 0
            for h in range(len(hiddenActivation)):
                sum_input += NeuralNets_global_data.W_ho[h][o] * hiddenActivation[h]
            # print outputActivation
            outputActivation[o] = sigmoid(sum_input)

    return [outputActivation,hiddenActivation]

def trainOneSample(input_sample, input_label):
################################################################
#
# This function performs the actual weight training (!!) of the
# network. It receives an input sample, and the correct label
# for the input sample (a number in [0,9] corresponding to
# which digit the input represents).
#
# You must use your FeedForward() function to compute the
# activation values for all the units in the network, then
# use the correct label to compute error values *for each
# output unit* (these will be returned!). Then you will
# adjust the weights using the error back-propagation method.
#
# Note that for networks with no hidden layer, you will be
# adjusting the weights in W_io and the computation is
# straightforward.
# For networks with a hidden layer, you will adjust weights
# in both W_ho and W_ih.
#
# NOTE: You will need to use the appropriate derivative of the
#       sigmoid function. Implement the function sigmoid_prime(a)
#       before you start working on this function, and mind
#       the fact that the sigmoid_prime function takes as
#       an argument the *activation* value for the unit, i.e.
#       the output of the neuron.
################################################################

	###############################################################
	# Use the 'errors' array to store the error between each
	# output neuron and the target value.
	# Error is defined as e=target - output
	#
		# We have one output neuron per digit. The 'target' output
		#   should be as follows:
		#      When using the logistic function as the activation
		#          - Correct output neuron should output .8
		#          - All others should output .2
		#      When using the hyperbolic tangent as the activation
		#	   - Correct output neuron should output .6
		#          - All other neurons should output -.6
		###############################################################

    errors=zeros(shape=(NeuralNets_global_data.N_out,1))

    ################################################################
    #
    # TO DO: Implement the backpropagation method for weight updates
    #        as discussed in lecture. Be careful to update the
    #        correct set of weights: W_io for networks with no
    #        hidden layer, and W_ih, W_ho for networks with
    #        a hidden layer.
################################################################
    result = FeedForward(input_sample)

    #Lists to store delta values
    delta_output = [0.0] * NeuralNets_global_data.N_out
    delta_hidden = [0.0] * NeuralNets_global_data.N_hidden

    #Set target value
    target = 0
    for o in range(NeuralNets_global_data.N_out):
        if NeuralNets_global_data.sig_type:
            if o == input_label:
                target = 0.6
            else:
                target = -0.6
        else:
            if o == input_label:
                target = 0.8
            else:
                target = 0.2
        errors[o] = (target - result[0][o])
        #Compute delta for output layer
        delta_output[o] = sigmoid_prime(result[0][o]) * (errors[o])
        # print "DELTA OUTPUT: h =  : " + str(delta_output[o])
    if NeuralNets_global_data.N_hidden == 0:
        #Update input -> output weights
        for i in range(len(NeuralNets_global_data.W_io)):
            for o in range(len(NeuralNets_global_data.W_io[i])):
                NeuralNets_global_data.W_io[i][o] += (NeuralNets_global_data.alpha *
                                                             (delta_output[o] * input_sample[i]))

    else:
        #Compute delta for hidden layer
        for h in range(NeuralNets_global_data.N_hidden):
            sum_dw = 0  #Sum of delta * associated weight
            for o in range(len(delta_output)):
                sum_dw += (delta_output[o] * NeuralNets_global_data.W_ho[h][o])
            delta_hidden[h] = (sigmoid_prime(result[1][h]) * sum_dw)

        #Update input -> hidden weights
        for i in range(NeuralNets_global_data.N_in):
            for h in range(NeuralNets_global_data.N_hidden):
                NeuralNets_global_data.W_ih[i][h] += (NeuralNets_global_data.alpha *
                                                      delta_hidden[h] * input_sample[i])
        #Update hidden -> output weights
        for h in range(NeuralNets_global_data.N_hidden):
            for o in range(NeuralNets_global_data.N_out):
                NeuralNets_global_data.W_ho[h][o] += (NeuralNets_global_data.alpha *
                                                     (delta_output[o] * result[1][h]))

    return(errors)
