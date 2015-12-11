import essentia
from essentia.standard import *
import essentia.streaming
import sys
from pylab import *
import matplotlib.pyplot as plt
import numpy as np

# def show_mfcc(audio_file):
# 	loader = essentia.standard.MonoLoader(filename = audio_file)
# 	audio = loader()
# 	
# 	plot(audio[1*44100:10*44100])
# 	show()
# 	
# 	w = Windowing(type = 'hann')
# 	spectrum = Spectrum()
# 	mfcc = MFCC()
# 	
# 	mfccs = []
# 	
# 	# COMPUTE MFCC's
# 	for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
# 		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
# 		mfccs.append(mfcc_coeffs)
# 	
# 	# Convert the list to an essentia.array
# 	mfccs = essentia.array(mfccs).T
# 	
# 	# plot
# 	plt.plot(mfccs)
# 	# imshow(mfccs[1:,:], aspect = 'auto')
# 	show()
# 	return
	
def meanMfcc(filename, outfile="key"):
	# Introducing the Pool: a good-for-all container
	#
	# A Pool can contain any type of values (easy in Python, not as much in C++ :-) )
	# They need to be given a name, which represent the full path to these values;
	# dot '.' characters are used as separators. You can think of it as a directory
	# tree, or as namespace(s) + local name.
	#
	# Examples of valid names are: bpm, lowlevel.mfcc, highlevel.genre.rock.probability, etc...

	# So let's redo the previous using a Pool
	
	loader = essentia.standard.MonoLoader(filename = filename)
	audio = loader()
	pool = essentia.Pool()
	
	w = Windowing(type = 'hann')
	spectrum = Spectrum()
	mfcc = MFCC()
	
	mfccs = []
	
	for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
		mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
		pool.add('lowlevel.mfcc', mfcc_coeffs)
		pool.add('lowlevel.mfcc_bands', mfcc_bands)
		
	# imshow(pool['lowlevel.mfcc'].T[1:,:], aspect = 'auto')
	# figure()
	
	# Let's plot mfcc bands on a log-scale so that the energy values will be better
	# differentiated by color
	# from matplotlib.colors import LogNorm
	# imshow(pool['lowlevel.mfcc_bands'].T, aspect = 'auto', interpolation = 'nearest', norm = LogNorm())
	# show()
	
	# <demo> --- stop ---
	# In essentia there is mostly 1 way to output your data in a file: the YamlOutput
	# although, as all of this is done in python, it should be pretty easy to output to
	# any type of data format.

	# output = YamlOutput(filename = 'mfcc.sig')
	# output(pool)
	
	# <demo> --- stop ---
	# Say we're not interested in all the MFCC frames, but just their mean & variance.
	# To this end, we have the PoolAggregator algorithm, that can do all sorts of
	# aggregation: mean, variance, min, max, etc...
	aggrPool = PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)
	
	#Write the mean mfcc to another pool
	meanMfccPool = essentia.Pool()
	meanMfccPool.add('mean_mfcc', aggrPool['lowlevel.mfcc.mean'])
	# print ("The pool size (aggr mean mfcc) is : " 
		# + str(size(meanMfccPool['mean_mfcc'])))
	output = YamlOutput(filename = 'mean_mfcc.sig')
	
	output(meanMfccPool)
	
	# with open('mean_mfcc.sig', 'r') as file:
	#     features = yaml.load(file)
	values = meanMfccPool['mean_mfcc']
	# print "Values for " + str(filename) + " : " + str(values)
	return values
	
def extractKey(filename, outfile="key"):
	# initialize algorithms we will use
	loader = MonoLoader(filename=filename)
	framecutter = FrameCutter()
	windowing = Windowing(type="blackmanharris62")
	spectrum = Spectrum()
	spectralpeaks = SpectralPeaks(orderBy="magnitude",
	                              magnitudeThreshold=1e-05,
	                              minFrequency=40,
	                              maxFrequency=5000, 
	                              maxPeaks=10000)
	hpcp = HPCP()
	key = Key()
	
	# use pool to store data
	pool = essentia.Pool() 
	
	# connect algorithms together
	loader.audio >> framecutter.signal
	framecutter.frame >> windowing.frame >> spectrum.frame
	spectrum.spectrum >> spectralpeaks.spectrum
	spectralpeaks.magnitudes >> hpcp.magnitudes
	spectralpeaks.frequencies >> hpcp.frequencies
	hpcp.hpcp >> key.pcp
	key.key >> (pool, 'tonal.key_key')
	key.scale >> (pool, 'tonal.key_scale')
	key.strength >> (pool, 'tonal.key_strength')
	
	# network is ready, run it
	essentia.run(loader)
	
	print pool['tonal.key_key'] + " " + pool['tonal.key_scale']
	
	# write to json file
	YamlOutput(filename=outfile, format="json")(pool)


# 
# # Use MonoLoader - returns down-mixed and resampled audio
# samples = ['samples/music-strings1.wav']
# 
# for filename in samples:
# 	mean_mfcc(filename)
