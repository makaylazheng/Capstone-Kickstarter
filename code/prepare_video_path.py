import numpy as np
import pandas as pd
import os

#%% read all the video paths
filepath = "kickstarter(total)"

video_folders = [
        "part1-1cover_page_img_video"
        , "part1-5update_page_img_video"
        , "part2-video"
    ]

video_paths = []
for i in range(len(video_folders)):
    folder_path = filepath + "/" + video_folders[i]
    folder_imgvideo = os.listdir(folder_path)
    for subfolder in folder_imgvideo:
        folder_path2 = folder_path + "/" + subfolder
        img_videos = os.listdir(folder_path2)
        for file_name in img_videos:
            if file_name[-3:] == "mp4":
                file_path = folder_path2 + "/" + file_name
                video_paths.append(file_path)
len(video_paths)
            
#%% Realpath and wrong video paths from Excel (266)

cover_excel = pd.read_excel("kickstarter(total)/1cover&2advertising_page.xlsx")

log = pd.DataFrame()
log["realpath"] = cover_excel.loc[:, "<realpath>"]
log["orgPath"] = cover_excel.loc[:,"介绍视频_video"].apply(lambda x: x[15:] if type(x)==str else x)  
def truncation_org(x):
    if pd.isnull(x) or x[-3:] != "mp4":
        return np.nan
    else:
        return x[33:]
log["orgPath_trun"] = log["orgPath"].apply(truncation_org)

truePath = pd.Series(index=log["realpath"], name="truePath")
for vp in video_paths:
    vp_trun1 = vp[89:]  # include 2 folder name
    slash_index = vp_trun1.index("/")
    vp_trun2 = vp_trun1[slash_index+1:]  # include 1 folder name
    vp_trun3 = vp_trun2[33:]  # truncated video name
    # first test
    if vp_trun2 in list(log["orgPath"]):
        vp_index = list(log["orgPath"]).index(vp_trun2)
        vp_realpath = log["realpath"][vp_index]
        truePath[vp_realpath] = vp
    # second test
    elif vp_trun3 in list(log["orgPath_trun"]):
        vp_index = list(log["orgPath_trun"]).index(vp_trun3)
        vp_realpath = log["realpath"][vp_index]
        truePath[vp_realpath] = vp
        
log = pd.merge(left=log, right=truePath, how="left", left_on="realpath", right_index=True)
log.to_csv("code/data/video_paths.csv", index=False)
