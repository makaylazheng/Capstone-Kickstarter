import pandas as pd
import os
import subprocess

#%% extract audio from video

video_paths = pd.read_csv("code/data/video_paths.csv", index_col=0)

for i in range(video_paths.shape[0]):
    # set input and output
    INPUT_VIDEO = video_paths["truePath"][i]
    if pd.isnull(INPUT_VIDEO) is False:
        relative_path_for_video = INPUT_VIDEO[89:]
        slash_index = relative_path_for_video.index("/")
        relative_path_for_video = relative_path_for_video[slash_index+1:][33:]
        relative_path_for_audio = relative_path_for_video[:-3] + "wav"
        OUTPUT_FILE = "code/data/audio/" + relative_path_for_audio
        # ensure output does not exist
        if os.path.isfile(OUTPUT_FILE) is True:
            os.remove(OUTPUT_FILE)
        # cmd
        cmd = "ffmpeg -i " + INPUT_VIDEO + " -ab 160k -ac 2 -ar 44100 -vn " + OUTPUT_FILE
        subprocess.call(cmd, shell=True)

#%% check for failed extraction and output audio paths

video_paths
video_paths["truePath"].count()
audio_folder = "code/data/audio/"
for file in os.listdir(audio_folder):
    key = file[:-3] + "mp4"
    file_index = list(video_paths["orgPath_trun"]).index(key)
    file_path = "code/data/audio/" + file
    video_paths.loc[file_index, "audio_path"] = file_path
video_paths["audio_path"].count()    
video_paths.to_csv("code/data/video_paths.csv")
