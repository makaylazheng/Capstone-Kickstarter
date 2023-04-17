import numpy as np
import pandas as pd

# read audio inference
audio_paths = np.load("code/speech-to-text/audio_paths.npy")
audio_texts = np.load("code/speech-to-text/audio_texts.npy")

src = pd.read_csv("code/data/video_paths.csv", index_col=0)
audio_texts_df = pd.DataFrame()
audio_texts_df["realpath"] = src["realpath"]
for i in range(len(audio_paths)):
    # match real path
    index = list(src["audio_path"]).index(audio_paths[i])
    # fill in dataframe
    audio_texts_df.loc[index, "audio_texts"] = audio_texts[i]
    
audio_texts_df.to_csv("code/speech-to-text/audio_texts_df.csv")
