# Capstone-Kickstarter

This is a capstone project to study the impacts of positive leadership on the crowdsourcing success on Kickstarter. The positive leadership is identified by some features of charismatic leadership extracted from online materials using machine learning methods. 

## Step 1
Some of the video paths are wrong, so we need to correct them and generate the new files recording all true paths. 

Code file: <code>00-prepare_video_path.py</code>

## Step 2
We start from the video analysis. The github repository we adopted to implement the emotion identification in videos is <https://github.com/hritikksingh/Twitter-video-emotion-and-sentiment-analysis>. The main code file we use for reference is <code>video_emotion.py</code>. The requirements for the environment is listed in the repository. 

Our code for this step is <code>analysis_video.py</code>. For each video, we generate a <code>csv</code> file including the scores for 7 emotions. Then, we integrate all <code>csv</code> files in the folder <code>video_emotion_csvs</code> into <code>analysis_video_results.csv</code>. 

## Step 3
The next step is to analyze the audio in the video. We need to first extract the audio from the video, and we take <https://github.com/HeZhang1994/video-audio-tools> as the reference. 

Then, we use two Python packages, <code>pyAudioAnalysis</code> (<https://github.com/tyiannak/pyAudioAnalysis>) and <code>Speech-Emotion-Recognition</code> (<https://github.com/Renovamen/Speech-Emotion-Recognition>). The corresponding requirements for both environments are listed in the repositories. 

## Step 4
The last type of materials left to be analyzed is texts. In addition to the texts in the campaign, we also take into account the texts in the video. We use the audio extracted in the last step and convert the speech into texts using the repository in <https://github.com/mozilla/DeepSpeech>. We first transform the speech to texts and then we organize the texts in the <code>csv</code> file. 

We use two dictionaries to analyze the texts, <code>Loughran-McDonald_MasterDictionary</code> and <code>NBF Dictionary</code> according to Hu and Ma, 2021. We clean the dictionaries by some NLP techniques in <code>dictionaries_clean.py</code>. After preprocessing the dictionaries, we implement the same NLP techniques to the textual materials and match to the words in dictionaries in <code>analysis_texts.py</code>. As such, we can get several features based on the number of matched words. 

## Step 5
After obtaining all the features we need, we begin to run regressions. We identify the dependent variables and regressors including independent variables and control variables first, then use different dependent variables to run regressions. The code is in <code>04-regression.py</code>. 