import essentia
import essentia.standard
from lxml import etree
import WS_global_data
from essentia.standard import *
from essentia.streaming import *
from pybrain.datasets import ClassificationDataSet
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
        # Create data set
        self.features_dataset = None
        # Alternative result - return a 3D array
        # 0 - file path
        # 1 - audio class
        # 2 - data
        self.features_array = [[] for i in range(3)]
        # self.features_array[0].append("ex")
        # self.features_array[0].append("test")
        # print self.features_array

    def get_file_dataset(self, path, audio_class, as_array=False):
        
        self.features_dataset = ClassificationDataSet(
                WS_global_data.N_input,
                nb_classes=WS_global_data.N_output,
                class_labels=WS_global_data.Classes)
                
        # Not directory - get features for single file
        self.addClassFeatures(audio_class, path,
                as_array=as_array, to_xml=False)    # Don't add to XML
        
        self.features_dataset._convertToOneOfMany()
        if as_array:
            return self.features_array
        return self.features_dataset  
          
    def get_dir_dataset(self, path, directory=True, as_array=False):
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
            features_dataset: The ClassificationDataSet containing a sample per
                              audio file.
        """

        self.features_dataset = ClassificationDataSet(
                WS_global_data.N_input,
                nb_classes=WS_global_data.N_output,
                class_labels=WS_global_data.Classes)
               
        # Get subfolders for provided directory
        class_dirs = os.listdir(directory)
        class_dir_paths = [""] * len(class_dirs)
        # Check if n. of classes matches global data
        if len(class_dirs) is not WS_global_data.N_output:
            print "ERROR: Sub-folder count does not \
                match number of audio classes"
            sys.exit()
        
        for i in range(len(class_dirs)):
            if WS_global_data.debug_extra:
                print "Entered class dir : " + class_dirs[i]
            class_dir_path = directory + class_dirs[i]
            if class_dirs[i] in WS_global_data.Classes:
                audio_class=WS_global_data.Class_indexes.get(class_dirs[i])
                # print str(class_dir_path) + "belongs to audio class: " + str(audio_class)
                self.addClassFeatures(audio_class, class_dir_path,
                        as_array=as_array, to_xml=True)
                
        print "Generated features_dataset with size " + str(len(self.features_dataset))
        
        # Save XML
        features_file=open(self.filename, 'w+')
        features_file.seek(0)
        features_file.write(etree.tostring(self.tree_root, pretty_print=True))
        features_file.truncate()
        
        if as_array:
            return self.features_array        

        self.features_dataset._convertToOneOfMany()
        return self.features_dataset

    def addClassFeatures(self, audio_class, path, as_array=False, to_xml=True):
        """
        Recursively traverse the class directory, adding any audio features to the
        data set.

        args:
            audio_class (int): The audio class index
            path: The directory to process
            features_dataset: The ClassificationDataSet
        """

        if os.path.isdir(path):
            audio_paths = os.listdir(path)
            # print "Audio paths: " + str(audio_paths)
            for i in range(len(audio_paths)):
                audio_full_path = path  + '/' + audio_paths[i]
                # print "Extacting features for " + audio_full_path + \
                #     ", class: " + str(audio_class)
                self.addClassFeatures(audio_class, audio_full_path, \
                    as_array=as_array)
                if WS_global_data.debug_extra:
                    print "Entered : " + audio_full_path
        else:
            features = self.extractFeatures(path, to_xml=to_xml)
            if WS_global_data.debug_extra:
                print "Features for " + path + " : " + str(features)
            if as_array:
                self.features_array[0].append(path)
                self.features_array[1].append(audio_class)
                self.features_array[2].append(features)
            else:
                self.features_dataset.addSample(features, audio_class)

            # print "Extracting features for " + path + "..."

    def extractFeatures(self, filepath, \
        mfcc=True, key_strength=True, spectral_flux=True, to_xml=False):
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
        features = {}
        result = None
        need_features = True
        missing_features = dict(zip(WS_global_data.feature_list, \
            (True for i in range(len(WS_global_data.feature_list)))))
        if len(self.tree_root):
            for feature_set in self.tree_root.iter():
                xml_features = []
                # Check if file already exists in XML, check for missing feats
                if feature_set.get('file') == filepath:
                    if WS_global_data.debug:
                        print "Reading feature set for " + filepath
                        print "Set : " + etree.tostring(feature_set)
                    for feature in feature_set.iter('feature'):
                        feature_name = feature.get('name')
                        if feature_name != None: # If valid feature
                            # Feature in XML that isn't in config...burn it!
                            if feature.get('name') not in WS_global_data.feature_list:
                                feature.getparent().remove(feature)
                                if WS_global_data.debug:
                                    print "Removed element " + \
                                        feature.get('name') + " from XML"
                            # Feature already in XML and matches global config
                            ## Add it to the result dictionary
                            else:
                                missing_features[feature_name] = False
                                xml_features.append(feature_name)
                                features[feature_name] = self.get_feature_values(feature)
                    # Find any features that aren't in XML but should be
                    for feature in WS_global_data.feature_list:
                        if feature not in xml_features:
                            missing_features[feature] = True
                    
                    print missing_features.values()
                    
                    # If all features are already present...simply get them!
                    if not all(val == True for val in missing_features.values()):
                        print "WHY"
                        need_features = False
                        result = self.getFeatureArrayFromXML(feature_set)
                    # Found the match, break out of the loop
                    break
                    
        # If any features are missing, get them and (otionally save to xml)
        if need_features:
            if WS_global_data.debug:
                print "Need additional features:"
            # Extract missing features, save to dictionary            
            for feat, is_needed in missing_features.iteritems():
                if is_needed:
                    if WS_global_data.debug:
                        print "Missing feature: " + feat + ". Extracting..."
                    features[feat] = extract_feature(feat, filepath)
            print features
            
            # Write feature set to XML tree
            if to_xml:
                feature_set = etree.SubElement(self.tree_root, 'feature-set', file=filepath)
                
                for name, result in features.iteritems():
                    print name,result
                    feature_name = name
                    feature_size = str(WS_global_data.feature_sizes.get(name))
                    tree_feature = etree.SubElement(feature_set, 'feature', \
                        name = feature_name, \
                        values = feature_size)
                    for v in xrange(len(result)):
                        if WS_global_data.debug_extra:
                            print "[INSERT_XML: " + str(result) + "]"
                        tree_val = etree.SubElement(tree_feature, 'value', \
                            index = str(v))
                        tree_val.text = str(result[v])
                        
                if __name__ == "__main__":
                    features_file=open(self.filename, 'w+')
                    features_file.write(etree.tostring(self.tree_root, pretty_print=True))
                    
            # Construct result array to be inserted into the dataset
            result = np.concatenate((
                        features.get(WS_global_data.feature_list[0]), \
                        features.get(WS_global_data.feature_list[1]), \
                        features.get(WS_global_data.feature_list[2]), \
                        features.get(WS_global_data.feature_list[3]), \
                        features.get(WS_global_data.feature_list[4]), \
                        features.get(WS_global_data.feature_list[5])),
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
                if feature.get('name') in WS_global_data.feature_list:
                    print "Feature: " + etree.tostring(feature)
                    for value in feature.iter('value'):
                        # print etree.tostring(value)
                        if result is None:
                            result = np.array([0.0] * WS_global_data.N_input)
                        # print "Value " + value.text
                        result[v] = float(value.text)
                        v+=1
                    feature_count+=1
        print "RESULT: " + str(result)
        return result.astype(float)
    
    def get_feature_values(self, feature):
        values = None
        v = 0
        for value in feature.iter('value'):
            # print etree.tostring(value)
            if values is None:
                values = np.array([0.0] * WS_global_data.feature_sizes[feature.get('name')])
            print "Value " + value.text
            values[v] = float(value.text)
            v+=1
        return np.array(values)

    # def show_mfcc(audio_file):
    #     loader = essentia.standard.MonoLoader(filename = audio_file)
    #     audio = loader()
    #
    #     plot(audio[1*44100:10*44100])
    #     show()
    #
    #     w = Windowing(type = 'hann')
    #     spectrum = Spectrum()
    #     mfcc = MFCC()
    #
    #     mfccs = []
    #
    #     # COMPUTE MFCC's
    #     for frame in FrameGenerator(audio, frameSize = 1024, hopSize = 512):
    #         mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
    #         mfccs.append(mfcc_coeffs)
    #
    #     # Convert the list to an essentia.array
    #     mfccs = essentia.array(mfccs).T
    #
    #     # plot
    #     plt.plot(mfccs)
    #     # imshow(mfccs[1:,:], aspect = 'auto')
    #     show()
    #     return


def extract_feature(function_name, filename):
    """
    Finds and calls the appropriate function matching the name of the feature
    requested.
    The function must be a global module function, not a class method.
    """
    try:
        func = globals()[function_name]
    except AttributeError:
        print 'Function ' + function_name + ' not found :('
    else:
        if WS_global_data.debug:
            print "Extracting " + function_name + " for " + filename
        return func(filename)

def mfcc(filename, outfile="key"):
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

def zcr(filename):
    loader = essentia.standard.MonoLoader(filename=filename)
    audio = loader()
    zcr = essentia.standard.ZeroCrossingRate()
    windowing = essentia.standard.Windowing(type="hann")
    spectrum = essentia.standard.Spectrum()

    pool = essentia.Pool()

    fluxArray = []

    zerocrossingrate = zcr(audio)
    # for frame in FrameGenerator(audio, frameSize = 2048, hopSize = 512):
    #     zerocrossingrate = zcr(spectrum(windowing(frame)))
    #     print "zerocrossingrate: " + str(zerocrossingrate)
    #     pool.add('lowlevel.zerocrossingrate', zerocrossingrate)

    aggrPool = essentia.standard.PoolAggregator(defaultStats = [ 'mean', 'var' ])(pool)

    #Write the mean mfcc to another pool
    meanZCRPool = essentia.Pool()
    meanZCRPool.add('zerocrossingrate', zerocrossingrate)

    # print "ZCR: " + str(meanZCRPool['zerocrossingrate'])

    avg_zcr = meanZCRPool['zerocrossingrate'][0]
    print "XCRRR: " + str(np.array(avg_zcr))
    return np.array([avg_zcr])

def key_strength(filename, outfile="key"):
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

    return np.array([pool['tonal.key_strength']])

    # # write to json file
    # YamlOutput(filename=outfile, format="json")(pool)

def spectral_flux(filename):
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
    return np.array([avg_flux])

def normalize(features):
    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(features)
    features = minmax_scaler.transform(features)
    features_norm = np.reshape(features, -1)
    return features_norm

def pitch_strength(filename):
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
    return np.array([avg_pitch_strength])

def lpc(filename):
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
    return np.array(avg_lpc)

if __name__ == "__main__":
    # extractKey("../samples/train/male voice/26456_186469-hq.mp3")
    # print "Guitar"
    # extractKey("../samples/train/guitar/30401_199517-hq.mp3")
    # print "Music: " +\
    #     str(lpc("../samples/train/music/electronic/331394_5403529-hq.mp3"))
    # print "Voice: " +\
    #     str(lpc("../samples/train/voice/male voice/21933_8043-hq.mp3"))
    # print "Ambient: " +\
    #     str(lpc("../samples/train/ambient/172130_3211085-hq.mp3")) + \
    #     str(lpc("../samples/train/ambient/city/152810_2366982-hq.mp3"))
    # print "Silence: " +\
    #     str(lpc("../samples/train/silence/61387_13258-hq.mp3"))
    ex = Extractor()
    ex.extractFeatures("../samples/train/music/piano/320148_140737-hq.mp3", save_xml=True)

#
# # Use MonoLoader - returns down-mixed and resampled audio
# samples = ['samples/music-strings1.wav']
#
# for filename in samples:
#     mean_mfcc(filename)
