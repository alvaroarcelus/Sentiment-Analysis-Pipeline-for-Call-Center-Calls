###################################
# AUDIO PREPROCESSING FUNCTIONS 
###################################

import pandas as pd 
import numpy as np 
import scipy
import os 
import soundfile as sf 
import subprocess
from pydub import AudioSegment
#moving files from directories
import shutil 
#Open AI's library 
import whisper
#regular expressions 
import re



# function that compresses wav file to mp3

def compress_wav_to_mp3(input_file, output_file, bitrate='128k'):
    try:
        # Load the WAV file using pydub
        audio = AudioSegment.from_wav(input_file)
        
        # Set the desired output bitrate (optional, default is 128k)
        audio.export(output_file, format="mp3", bitrate=bitrate)
        
        print("Conversion successful. MP3 file saved as:", output_file)
    except Exception as e:
        print("Error during conversion:", e)



"""
This function changes de audio file's:
    1) Corrects wav format to compatible one
    2) Compresses aufio file from wav to mp3 (x10 smaller)
    3) Changes file name to one without operators
"""
def change_format(file_path, saving_path):
    directory, filename = os.path.split(file_path)

    #extract name from path 
    original_name = os.path.basename(file_path)
    new_name = os.path.splitext(original_name)[0] + '.mp3'
    new_file_path = os.path.join(directory, new_name)

    #change from wav to mp3
    os.rename(file_path, new_file_path)
    #change from mp3 to wav
    subprocess.call(['ffmpeg', '-i', new_file_path, file_path])

    #delete mp3 version
    os.remove(new_file_path)

    #compress wav file to mp3
    mp3_save_name = re.sub(r"\.wav$", ".mp3", file_path)
    compress_wav_to_mp3(file_path, mp3_save_name)

    #save file in it's own folder (saving_folder)
    shutil.move(mp3_save_name, saving_path)

    #rename the file as algorithms wont work with operators such as plus sign
    email = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', os.path.basename(mp3_save_name)).group(0)
    print(email)
    os.rename(saving_path + os.path.basename(mp3_save_name), saving_path + email + '.mp3')


#Function that trims an audio starting based on an interval represented in seconds 
# audio, sample_rate = sf.read('myRealdeal.mp3')

def trim_me(init_time, end_time, sample_rate, audio):
    start_trim = int(init_time*sample_rate)
    end_trim = int(end_time*sample_rate)
    #trimming vector
    trimmed = audio[start_trim:end_trim]
    return trimmed 



#Function that merges all the audio of a single speaker 
def  cluster_me(speaker, audio_file, audio_intervals): 
    #filtering only speaker 0 data
    spk_df = audio_intervals[audio_intervals.speaker == speaker].reset_index(drop=True)
    #transforming audio file into a vector
    audio, spl_rate = sf.read(audio_file) 

    #creating vectors with all initiating and ending values
    init_intervals = spk_df.init_time
    ending_intervals = spk_df.end_time

    final_trim = []

    #selecting audio from all the intervals where the speaker interacted
    for i in range(len(spk_df)):
        u = float(init_intervals[i]) #tranforming from object to float
        k = float(ending_intervals[i])

        #trimming audios with previous function
        audio_new = trim_me(u,k, spl_rate, audio)
        
        #append results  
        for v in audio_new: 
            final_trim.append(v)

    #exporting trimmed audio file
    sf.write(speaker + '.wav', final_trim, spl_rate)


#Diarization function (Speaker segmentation)
#When calling it, we are crecating two seperate audios

#we need to enable developer mode on Windows in order for it to function properly
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token="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")


def diarize_me(audio):
    diarization = pipeline(audio , num_speakers=2)

    #from rttm file to dataframe 
    with open("audio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)

    rttm_file = open("audio.rttm", "r")
    rttm_lines = rttm_file.readlines()
    rttm_file.close()


    speaker = []
    init_time=[]
    duration=[]
    #extract relevant fields 
    for line in rttm_lines:
        fields = line.strip().split(" ")
        speaker.append(fields[7])
        init_time.append(fields[3])
        duration.append(fields[4])

    #create pandas dataframe 
    intervals = pd.DataFrame({'speaker':speaker, 'init_time': init_time, 'duration':duration})

    #change data types 
    intervals['init_time'] = intervals['init_time'].astype(float)
    intervals['duration'] = intervals['duration'].astype(float)

    #add new column with end_time
    intervals['end_time'] = intervals.init_time + intervals.duration

    #remove rttm file from environment 
    os.remove('audio.rttm')

    #trimming audio for speaker 1 
    cluster_me('SPEAKER_00', audio, intervals)

    #trimming audio for speaker 2
    cluster_me('SPEAKER_01', audio, intervals)
    

#function that transcribes an audio
model = whisper.load_model('medium') 

def transcribe_me(audio):
    result = model.transcribe(audio, language='es', fp16=False)
    return result['text']


#Function that merges all the data into a single dataframe 

def collect_data(data,recording_folder, trans_0, trans_1):
    #find the employee's email 
    files = os.listdir(recording_folder)
    selected_file = None

    for file in files:
        if "@" in file:
            email = file
            #removing .wav extension from filename 
            email = re.sub(r"\.wav$", "", email)
            break

    #append new entry to global dataframe
    new_entry = pd.DataFrame({'email':[email], 'client':[trans_0], 'employee':[trans_1]})
    data = pd.concat([data, new_entry])
    return data

#Function that creates individual folders for every single file that is inside a source directory
def create_folders(output_directory, source_directory):
    #calculate number of files inside folder
    num_files = len(os.listdir(source_directory))
    print('Files to be processed: '+str(num_files))

    #creating folders where each pair of audios will be stored
    for i in range(1, num_files + 1):
        folder_name = f"Recording_{i}"
        folder_path = os.path.join(output_directory, folder_name)
        os.makedirs(folder_path)

    print('---------'*5)
    print(str(num_files) + ' folders were created succesfully')
    print('---------'*5)


# Function that creates individual folders given the number of files inside the specified path (the function's sole parameter)
def create_folders(source_directory):
    #determine folder where audios will be saved
    output_directory = 'Segmented Audios'

    #calculate number of files inside folder
    num_files = len(os.listdir(source_directory))
    print('Files to be processed: '+str(num_files))

    #creating folders where each pair of audios will be stored
    for i in range(1, num_files + 1):
        folder_name = f"Recording_{i}"
        folder_path = os.path.join(output_directory, folder_name)
        os.makedirs(folder_path)

    print('---------'*5)
    print(str(num_files) + ' folders were created succesfully')
    print('---------'*5)

    #returning the number of files to be processed ~ num of folders created
    return num_files






