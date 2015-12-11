import os, errno, json
import freesound
import sys

#Freesound.org API key - passed as a token with every request
API_KEY = "518e7f5dcf83e2551abbb2f1a8d87bc4945dcb6f";

descriptors = "lowlevel.spectral_centroid," + \
            "lowlevel.mfcc.mean,lowlevel.pitch_salience," + \
            "lowlevel.silence_rate_20dB," + \
            "tonal.chords_count," + \
            "tonal.key_strength"
            

filter_param = "samplerate: 44100 channels: 2 duration: [5 TO 10]"

class Downloader:
    def __init__(self):
        # Get the freesound client and set the api token
        self.c = freesound.FreesoundClient()
        self.c.set_token(API_KEY, "token")
        
        # Number of results per page
        self.results_limit = 20
        # Whether descriptors should be normalized
        self.normalized = 1        
        # Possible fields to return:
        ## name
        ## avg_rating
        ## license
        ## analysis
        ## descriptors=[lowlevel.spectral_centroid, lowlevel.mfcc.mean]
        self.fields = "id,name,description,previews,username,analysis"
        
        
        
    """
    Method to retrieve the list of sounds queried using the freesound apiv2
    """
    def get_sounds(self, query="", tag=None, durationFrom=5, durationTo=10, results=5):
        # Text search returns an interable Pager object
        queryText = query + "&page_size=" + str(self.results_limit) +\
            "&normalized=" + str(self.normalized)
        
        filterText = "samplerate: 44100 channels: 2 duration: [" + \
            str(durationFrom) + " TO " + str(durationTo) + "]"
        
        fieldsText = self.fields
        
        results_pager = self.c.text_search(query=queryText, filter=filterText, \
            fields=fieldsText)
            
        # Array to store the Sound objects
        sounds = []
        print "Query: \"" + keyword + "\""
        for i in range(len(results_pager.results)):
            #Get the current item
            result = results_pager[i]
            description = result.description
            # print result.__dict__
            #Create a directory to save the audio and descriptors
            path = "../samples/train/" + query + "/" + result.name
            # print "Previews:" ,result.previews.preview_hq_mp3
            # print "Username: ", result.username
            # print "Analysis: " , result.analysis, "\n"
            
            # Save the previews in the sounds/ directory using the sound filename            
            print "\n- Downloading sound " + str(i + 1) + " of " + str(len(results_pager.results)) + " with filename: " + result.name + \
                        "\n  Description: " + result.description[0:30] + "..."
            result.retrieve_preview("../samples/train/" + query)
            
            # Get the sound features and save them to a JSON file
            # features = result.get_analysis(descriptors=descriptors)
            # print features
            # export_json_features(features, result.name, path)
            
def create_path(keyword):
    try:
        os.makedirs(keyword)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise        
            
def export_json_features(dict, name, path):
    with(open(path + name + ".json", 'w+')) as json_file:
        # json_data = {
        #     'lowlevel.mfcc.mean': dict.lowlevel.mfcc.mean,
        #     'lowlevel.spectral_centroid': dict.lowlevel.spectral_centroid,
        #     'lowlevel.mfcc.mean': dict.lowlevel.mfcc.mean,
        #     'lowlevel.pitch_salience': dict.lowlevel.pitch_salience,
        #     'lowlevel.silence_rate_20dB': dict.lowlevel.silence_rate_20dB,
        #     'tonal.chords_count': dict.tonal.chords_count,
        #     'tonal.key_strength': dict.tonal.key_strength
        # }
        json.dump(dict, json_file)
            
#----------------------------MAIN PROGRAM--------------------------
if __name__ == "__main__":
    #Validate arguments
    if len(sys.argv) != 2:
        print "Incorrect usage. Please provide keyword as argument: " +\
        "\npython WhatSound_download.py <keyword>"
        sys.exit()
    else:   #Parse the argument
        keyword = sys.argv[1]   
        #Create directory to save the previews in
        create_path("../samples/train/" + keyword)
        #Download sounds 
        downloader = Downloader();    
        downloader.get_sounds(query=keyword)
