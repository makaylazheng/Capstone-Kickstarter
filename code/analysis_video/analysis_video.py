import numpy as np
import cv2
from fastai import *
from fastai.vision import *
import pandas as pd
from imutils.video import FileVideoStream
import time
import dlib
import openpyxl
import glob
from os.path import join

#%% analyze every video and save a csv for each of them

root = ''
# read video paths
video_paths_csv = pd.read_csv(
    join(root, 'code', 'data', 'video_paths.csv')
    , index_col=0
    )

# function for analyzing a single video file
def run_video(video_path):
    if video_path != video_path:
        return
    #Loading Model files
    vidcap = FileVideoStream(video_path).start()
    # load models
    learn = load_learner(
        join(root, 'code', 'analysis_video')
        , 'export.pkl'
        )
    face_cascade = cv2.CascadeClassifier(
        join(root, 'code', 'analysis_video', 'haarcascade_frontalface_default.xml')
        )
    predictor = dlib.shape_predictor(
        join(root, 'code', 'analysis_video', 'shape_predictor_68_face_landmarks.dat')
        )
    framecount = 0
    data=[]
    while vidcap.more():
        frame = vidcap.read()
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coord = face_cascade.detectMultiScale(gray, 1.1, 20, minSize=(30, 30))
        for coords in face_coord:
            X, Y, w, h = coords
            H, W, _ = frame.shape
            X_1, X_2 = (max(0, X - int(w * 0.3)), min(X + int(1.3 * w), W))
            Y_1, Y_2 = (max(0, Y - int(0.3 * h)), min(Y + int(1.3 * h), H))
            img_cp = gray[Y_1:Y_2, X_1:X_2].copy()
            if framecount % 10 == 0:
                prediction, idx, probability = learn.predict(Image(pil2tensor(img_cp, np.float32).div_(225)))
                data.append([prediction])
        framecount += 1
    df = pd.DataFrame(data, columns = [ 'Expression'])
    vidcap.stop()
    return df

for i in range(video_paths_csv.shape[0]):
    print(i)
    video_path = video_paths_csv.loc[i, 'truePath']
    df = run_video(video_path)
    if df is None:
        continue
    video_name = video_paths_csv.loc[i, 'orgPath_trun']
    csv_name = video_name[:-3] + 'csv'
    df.to_csv(join(root, 'code', 'analysis_video', 'video_emotion_csvs', csv_name))

#%% calculate the emotion percentage for each video

csv_root = join(root, 'code', 'analysis_video', 'video_emotion_csvs')
csv_list = os.listdir(csv_root)

# function to calculate the percentage given a csv
def get_perc(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    count={}
    for i in range(df.shape[0]):
        cell_val = df.iloc[i, 0]
        if cell_val not in count:
            count[cell_val] = 1
        else:
            count[cell_val] += 1
    total = df.shape[0]
    for key, value in count.items():
        count[key] = (value / total) * 100
    return count

analysis_video_results = pd.DataFrame()
analysis_video_results['realpath'] = video_paths_csv['realpath']
for i in range(len(csv_list)):
    # get percentages
    csv_name = csv_list[i]
    csv_path = join(csv_root, csv_name)
    count = get_perc(csv_path)
    # match realpath
    video_name = csv_name[:-3] + 'mp4'
    index = list(video_paths_csv['orgPath_trun']).index(video_name)
    for key, value in count.items():
        analysis_video_results.loc[index, key] = count[key]
analysis_video_results.to_csv(join(root, 'code', 'analysis_video', 'analysis_video_results.csv'))
    
