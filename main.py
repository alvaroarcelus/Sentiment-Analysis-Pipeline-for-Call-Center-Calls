#################################
# Libraries and packages
#################################
import pandas as pd 
import numpy as np 
import scipy
import os 
import soundfile as sf 
import subprocess
import shutil 
import re
import whisper
import datetime
from tqdm import tqdm # progress bar
#importing local "in-house" functions for data preprocessing 
import functions.preprocessing as prs
import functions.myTransformers as trans
import os

# Change the working directory for Ubuntu server
# new_path = '/home/ubuntu/NLP_Production'
# os.chdir(new_path)

# delete all files inside Segmented Audios directory 
# precautionary measure in case the code was ran before
folder_path = 'Segmented Audios'
for root, dirs, files in os.walk(folder_path, topdown=False):
    for dir in dirs:
        folder_to_remove = os.path.join(root, dir)
        shutil.rmtree(folder_to_remove)


#################################
# Data Preprocessing Pipeline
#################################

def audio_process_pipeline(audio_source_path):

    #create folders where processed data will be stored 
    num_files = prs.create_folders(audio_source_path)

    data = pd.DataFrame({'email':[], 'client':[], 'employee':[]})

    # Iterate over all files in the folder
    for root, dirs, files in os.walk(audio_source_path):
        for file_name, i in zip(tqdm(files, desc="Processing", ncols=100), range(1, num_files + 1)):
            file_path = os.path.join(root, file_name)
                   
            # ----> TRANSFORM FORMAT 
            print('Transforming Format')
            folder_name = f"Recording_{i}"
            save_path = 'Segmented Audios/' + folder_name + '/'
            prs.change_format(file_path, save_path)
            print('Format Transformed')

            # ----> APPLY DIARIZATION
            print('Diarization Initialized')
            target_audio = save_path + '/' + os.listdir(save_path)[0]  #selecting audio inside target folder 
            prs.diarize_me(target_audio)
            print('Diarization complete')
            
            # ----> SAVE BOTH AUDIOS
            shutil.move('SPEAKER_00.wav', save_path + '/')
            shutil.move('SPEAKER_01.wav', save_path + '/')

            # ----> TRANSCRIBE 
            spk0 = prs.transcribe_me(save_path + '/SPEAKER_00.wav')
            spk1 = prs.transcribe_me(save_path + '/SPEAKER_01.wav')

            # ----> APPEND DATA
            #create a dataframe for storing all conversations 
            data = prs.collect_data(data ,save_path + '/', spk0, spk1)
            print('-----------------'*3)
            print('Audio number '+str(i)+' has been processed correctly')
            print('-----------------'*3)

    return data 

#calling main function 
res = audio_process_pipeline('recordings/')

current_datetime = datetime.datetime.now() # retrieving current date
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
file_name = f"transcriptions/transcription_{formatted_datetime}.csv"
res.to_csv(file_name, index=False)

#################################
# Sentiment Analysis Pipeline
#################################

#retrieving transcripts from current day 
df = pd.read_csv(file_name)
df = df.fillna(" ")

print('-----------------'*3)
print('Sentiment Analysis Pipeline: initialized')
print('-----------------'*3)

scores = trans.sentiment_pipeline(df)

print('-----------------'*3)
print('Sentiment Analysis Pipeline: processed correctly')
print('-----------------'*3)

# Exporting resuls as a .csv file 
current_datetime = datetime.datetime.now() # retrieving current date
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
file_name = f"scores/scores_{formatted_datetime}.csv"
scores.to_csv(file_name, index=False)

