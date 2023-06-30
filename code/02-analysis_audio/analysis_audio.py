import os
from os.path import join
from pyAudioAnalysis import audioTrainTest as aT
from speechemotionrecognition.dnn import LSTM   
from pandas import DataFrame as df
import pandas as pd
import numpy as np
from pydub import AudioSegment
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc

#%% audio analysis by pyAudioAnalysis: arousal and valence

root = "code"
audio_root = root + "/02-analysis_audio/audio"
model_path = root + "/02-analysis_audio/pyAudioAnalysis_model"
pyAudioAnalysisModel = model_path + '/svmSpeechEmotion'
audios = os.listdir(audio_root)

scr = pd.read_csv("code/video_audio_paths.csv" ,index_col=0)
audio_result_df = df()
audio_result_df["realpath"] = scr["realpath"]
for i in range(len(audios)):
    print(i)
    complete_path = audio_root + "/" + audios[i]
    # match with realpath
    index = list(scr["audio_path"]).index(complete_path)
    # Get length
    audio_seg = AudioSegment.from_file(complete_path)
    audio_length = audio_seg.duration_seconds
    # Get vocal arousal and valence
    audio_emotion = aT.file_regression(complete_path, pyAudioAnalysisModel, 'svm') 
    audio_result_df.loc[index, 'Onset'] = round(0, 2)
    audio_result_df.loc[index, 'Offset'] = round(audio_length, 2)
    audio_result_df.loc[index, 'Duration'] = audio_result_df.loc[index, 'Offset'] - audio_result_df.loc[index, 'Onset']
    audio_result_df.loc[index, 'Vocal-Arousal'] = audio_emotion[0][0]
    audio_result_df.loc[index, 'Vocal-Valence'] = audio_emotion[0][1]

#%% audio analysis by speechemotionrecognition

root = "code"
audio_root = root + "/02-analysis_audio/audio"
model_path = root + "/02-analysis_audio/speechemotionrecognition_model"
speechemotionrecognitionModel = model_path + '/best_model_LSTM_39.h5'
audios = os.listdir(audio_root)

scr = pd.read_csv("code/video_audio_paths.csv" ,index_col=0)
for i in range(len(audios)):
    print(i)
    complete_path = audio_root + "/" + audios[i]
    # match with realpath
    index = list(scr["audio_path"]).index(complete_path)
    # Get length
    audio_seg = AudioSegment.from_file(complete_path)
    audio_length = audio_seg.duration_seconds
    # Extract audio feature
    audio_seg_channel_list = audio_seg.split_to_mono()
    with open(complete_path, 'wb') as f:
        audio_seg_channel_list[0].set_frame_rate(16000).export(f, format='wav')
    audio_feature = get_feature_vector_from_mfcc(complete_path, flatten=False)
    audio_feature = np.array([audio_feature])
    # Get positive and negative
    lstm_model = LSTM(input_shape=(198, 39), num_classes=4)
    lstm_model.restore_model(speechemotionrecognitionModel)
    audio_emotion = list(lstm_model.model.predict(audio_feature)[0])
    if audio_result_df.loc[index, 'Onset'] != round(0, 2):
        print(index, "Onset conflict")
    if audio_result_df.loc[index, 'Offset'] != round(audio_length, 2):
        print(index, "Offset conflict")
    audio_result_df.loc[index, 'Vocal-Positive'] = audio_emotion[2]
    audio_result_df.loc[index, 'Vocal-Negative'] = audio_emotion[3]
    
audio_result_df.to_csv("code/02-analysis_audio/analysis_audio_results.csv"
                       , float_format="%.2f")
