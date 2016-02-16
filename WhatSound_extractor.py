import essentia
import essentia.standard
from lxml import etree
import WhatSound_global_data
from essentia.standard import *
from essentia.streaming import *
from pybrain.datasets            import ClassificationDataSet
import sys
from pylab import *
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import os

class Extractor():
	
	def __init__(self):
		# XML tree root element
		self.tree_root = etree.Element('feature-list')
		self.filename = 'wsfeatures.xml'
		self.new_file = True
		if os.path.isfile(self.filename):
			self.new_file = False
			tree_wrapper = etree.parse(self.filename)
			self.tree_root = tree_wrapper.getroot()
				
		#Create data set
		self.dataset = None
			
	def getFeatureSet(self, directory, **kwargs):
		"""
		Generate a ClassificationDataSet for the provided feature set. The directory
		will be traversed depth-first until an audio file is found.

		args:
			directory (string): Training or testing directory.
					The top-level directory containing 4 sub-directories
					corresponding to the audio classes. Each sub-folder
					can contain other folders and/or audio files.
		kwargs:
			Toggle features from the global feature list on/off
		Returns:
			dataset: The ClassificationDataSet containing a sample per audio file.
		"""
		
		self.dataset = ClassificationDataSet(WhatSound_global_data.N_input, \
			nb_classes=WhatSound_global_data.N_output, \
			class_labels=WhatSound_global_data.Classes)
			
		#Get subfolders for provided directory
		class_dirs = os.listdir(directory)
		class_dir_paths = [""] * len(class_dirs)
		#Check if n. of classes matches global data
		if len(class_dirs) is not WhatSound_global_data.N_output:
			print "ERROR: Sub-folder count does not match number of audio classes"
			sys.exit()
			
		for i in range(len(class_dirs)):
			class_dir_path = directory + class_dirs[i]
			if WhatSound_global_data.Class_indexes.has_key(class_dirs[i]):
				audio_class = WhatSound_global_data.Class_indexes.get(class_dirs[i])
				# print str(class_dir_path) + "belongs to audio class: " + str(audio_class)
				self.addClassFeatures(audio_class, class_dir_path)
		print "Generated dataset with size " + str(len(self.dataset))
		# print "DATASET: " + str(self.dataset)
		features_file = open(self.filename, 'w+')
		features_file.write(etree.tostring(self.tree_root, pretty_print=True))
		
		self.dataset._convertToOneOfMany()
		return self.dataset
	
	def addClassFeatures(self, audio_class, path):
		"""
		Recursively traverse the class directory, adding any audio features to the
		data set.
		
		args:
			audio_class (int): The audio class index
			path: The directory to process
			dataset: The ClassificationDataSet
		"""
		
		if os.path.isdir(path):
			audio_paths = os.listdir(path)
			# print "Audio paths: " + str(audio_paths)
			for i in range(len(audio_paths)):
				audio_full_path = path  + '/' + audio_paths[i]
				# print "Extacting features for " + audio_full_path + \
				# 	", class: " + str(audio_class)
				self.addClassFeatures(audio_class, audio_full_path)
		else:
			self.dataset.addSample(self.extractFeatures(path), audio_class)
			# print "Extracting features for " + path + "..."
			
	def extractFeatures(self, filepath, \
		mfcc=True, key_strength=True, spectral_flux=True, save_xml=False):
		"""
		Extract features for the provided audio file.
		
		Args:
			filename: The audio file used for feature extraction.
		kwargs:
			mfcc (bool): Toggle mel-freq-spectrum-coefficients
			key_strength (bool): Toogle key 'strength' eg. F# major with 0.36 strength
			spectral_flux (bool): Toggle avg flux for this audio file.
		Returns:
			result: The feature vector.
		"""
		# Check is features for this audio file already exist in XML
		result = None
		exists = False
		if len(self.tree_root):
			for feature_set in self.tree_root.iter():
				#If feature set element exists - generate np array from xml
				if feature_set.get('file') == filepath:	
					exists = True
					result = self.getFeatureArrayFromXML(feature_set)
					break	# Already exists in XML tree, dont add
		if not exists:
			# Extract all features, save to numpy array
			features = {}
			features[WhatSound_global_data.feature_list[0]] = meanMfcc(filepath)
			features[WhatSound_global_data.feature_list[1]] = np.array([extractKey(filepath)])
			features[WhatSound_global_data.feature_list[2]] = np.array([extractSpectralFlux(filepath)])
			features[WhatSound_global_data.feature_list[3]] = np.array([extractZCR(filepath)])
			features[WhatSound_global_data.feature_list[4]] = np.array([extractPitchStrength(filepath)])
			features[WhatSound_global_data.feature_list[5]] = extractLPC(filepath)
			
			
			print features
			# Write feature set to XML tree
			feature_set = etree.SubElement(self.tree_root, 'feature-set', file=filepath)
			# Add features
			
			for i in xrange(len(features)):
				feature_name = str(WhatSound_global_data.feature_list[i])
				feature_size = str(WhatSound_global_data.feature_sizes.get(feature_name))
				tree_feature = etree.SubElement(feature_set, 'feature', \
					name=feature_name, \
					values=feature_size)
				for v in xrange(len(features[WhatSound_global_data.feature_list[i]])):
					tree_val = etree.SubElement(tree_feature, 'value', \
						index=str(v))
					tree_val.text = str(features[WhatSound_global_data.feature_list[i]][v])
			
			result = np.concatenate((
						features.get(WhatSound_global_data.feature_list[0]), \
	 					features.get(WhatSound_global_data.feature_list[1]), \
						features.get(WhatSound_global_data.feature_list[2]), \
						features.get(WhatSound_global_data.feature_list[3]), \
						features.get(WhatSound_global_data.feature_list[4]), \
						features.get(WhatSound_global_data.feature_list[5])),
						axis=0)
			
		return result
		
	def getFeatureArrayFromXML(self, feature_set):
		result = None
		# If not empty - generate array
		if len(feature_set):
			# Note: iterating doesn't work (mismatch of values)
			# need to index through all values
			feature_count = 0
			v = 0
			for feature in feature_set.iter('feature'):
				if feature.get('name') == \
					WhatSound_global_data.feature_list[feature_count]:
					# print "Feature: " + etree.tostring(feature)
					for value in feature.iter('value'):
						# print etree.tostring(value)
						if result is None:
							result = np.array([0.0] * WhatSound_global_data.N_input)	
						# print "Value " + value.text
						result[v] = float(value.text)
						v+=1
					feature_count+=1
		print "RESULT: " + str(result)
		return result.astype(float)
		

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

	w = essentia.standard.Windowing(type = 'hann')
	spectrum = essentia.standard.Spectrum()
	mfcc = essentia.standard.MFCC()

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
	aggrPool = essentia.standard.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

	#Write the mean mfcc to another pool
	meanMfccPool = essentia.Pool()
	meanMfccPool.add('mean_mfcc', aggrPool['lowlevel.mfcc.mean'])
	# print ("The pool size (aggr mean mfcc) is : "
		# + str(size(meanMfccPool['mean_mfcc'])))
	# output = YamlOutput()
	# output(filename = 'mean_mfcc.sig')
	#
	# output(meanMfccPool)

	# with open('mean_mfcc.sig', 'r') as file:
	#     features = yaml.load(file)
	values = np.array(meanMfccPool['mean_mfcc'][0])
	values = np.reshape(values, (-1, 1))
	# print "Values for " + str(filename) + " : " + str(values)
	values_norm = normalize(values)
	return values_norm

def extractZCR(filename):
	loader = essentia.standard.MonoLoader(filename=filename)
	audio = loader()
	zcr = essentia.standard.ZeroCrossingRate()
	windowing = essentia.standard.Windowing(type="hann")
	spectrum = essentia.standard.Spectrum()

	pool = essentia.Pool()

	fluxArray = []

	zerocrossingrate = zcr(audio)
	# for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
	# 	zerocrossingrate = zcr(spectrum(windowing(frame)))
	# 	print "zerocrossingrate: " + str(zerocrossingrate)
	# 	pool.add('lowlevel.zerocrossingrate', zerocrossingrate)

	aggrPool = essentia.standard.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

	#Write the mean mfcc to another pool
	meanZCRPool = essentia.Pool()
	meanZCRPool.add('zerocrossingrate', zerocrossingrate)

	# print "ZCR: " + str(meanZCRPool['zerocrossingrate'])

	avg_zcr = meanZCRPool['zerocrossingrate'][0]
	return avg_zcr

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

	# print str(pool['tonal.key_key']), \
		# str(pool['tonal.key_scale']),  "Key strength: " ,  str(pool['tonal.key_strength'])

	return pool['tonal.key_strength']

	# # write to json file
	# YamlOutput(filename=outfile, format="json")(pool)

def extractSpectralFlux(filename):
	loader = essentia.standard.MonoLoader(filename=filename)
	audio = loader()
	flux = essentia.standard.Flux()
	windowing = essentia.standard.Windowing(type="blackmanharris62")
	spectrum = essentia.standard.Spectrum()

	pool = essentia.Pool()

	fluxArray = []

	for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
		flux_val = flux(spectrum(windowing(frame)))
		pool.add('lowlevel.flux', flux_val)

	aggrPool = essentia.standard.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

	#Write the mean mfcc to another pool
	meanFluxPool = essentia.Pool()
	meanFluxPool.add('mean_flux', aggrPool['lowlevel.flux.mean'])

	# print meanFluxPool['mean_flux']

	avg_flux = meanFluxPool['mean_flux'][0]
	return avg_flux

def normalize(features):
	minmax_scaler = MinMaxScaler()
	minmax_scaler.fit(features)
	features = minmax_scaler.transform(features)
	features_norm = np.reshape(features, -1)
	return features_norm

def extractPitchStrength(filename):
	loader = essentia.standard.MonoLoader(filename=filename)
	audio = loader()
	pitch_sal = essentia.standard.PitchSalience()
	windowing = essentia.standard.Windowing(type="blackmanharris62")
	spectrum = essentia.standard.Spectrum()

	pool = essentia.Pool()

	fluxArray = []

	for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
		pitch_sal_val = pitch_sal(spectrum(windowing(frame)))
		pool.add('lowlevel.pitch_strength', pitch_sal_val)

	aggrPool = essentia.standard.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

	#Write the mean mfcc to another pool
	mean_ps_pool = essentia.Pool()
	mean_ps_pool.add('mean_pitch_strength', aggrPool['lowlevel.pitch_strength.mean'])

	# print meanFluxPool['mean_flux']

	avg_pitch_strength = mean_ps_pool['mean_pitch_strength'][0]
	return avg_pitch_strength
	
def extractLPC(filename):
	loader = essentia.standard.MonoLoader(filename=filename)
	audio = loader()
	lpc = essentia.standard.LPC()
	windowing = essentia.standard.Windowing(type="hann")
	spectrum = essentia.standard.Spectrum()

	pool = essentia.Pool()

	fluxArray = []

	for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
		lpc_coeff, refl = lpc(spectrum(windowing(frame)))
		# print lpc_coeff, refl
		pool.add('lowlevel.lpc', lpc_coeff)		
		pool.add('lowlevel.lpc_reflection', refl)
		
	print pool.__dict__

	aggrPool = essentia.standard.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

	#Write the mean mfcc to another pool
	mean_lpc_pool = essentia.Pool()
	mean_lpc_pool.add('mean_lpc', aggrPool['lowlevel.lpc.mean'])

	# print meanFluxPool['mean_flux']

	avg_lpc = mean_lpc_pool['mean_lpc'][0]
	return avg_lpc

if __name__ == "__main__":
	# extractKey("../samples/train/male voice/26456_186469-hq.mp3")
	# print "Guitar"
	# extractKey("../samples/train/guitar/30401_199517-hq.mp3")
	print "Music: " +\
		str(extractLPC("../samples/train/music/electronic/331394_5403529-hq.mp3"))
	print "Voice: " +\
		str(extractLPC("../samples/train/voice/male voice/21933_8043-hq.mp3"))
	print "Ambient: " +\
		str(extractLPC("../samples/train/ambient/172130_3211085-hq.mp3")) + \
		str(extractLPC("../samples/train/ambient/city/152810_2366982-hq.mp3"))
	print "Silence: " +\
		str(extractLPC("../samples/train/silence/61387_13258-hq.mp3"))

#
# # Use MonoLoader - returns down-mixed and resampled audio
# samples = ['samples/music-strings1.wav']
#
# for filename in samples:
# 	mean_mfcc(filename)
