# get the .pbmm and .scorer files at https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
import os
import wavTranscriber
import numpy as np

# model root
dirName = "code\\speech-to-text"
# audio root
audio_root = "code\\data\\audio\\"
# load pretrained model
output_graph, scorer = wavTranscriber.resolve_models(dirName)
model_retval = wavTranscriber.load_model(output_graph, scorer)

# use two lists to store path and texts
audio_paths = []
audio_texts = []
audios = os.listdir(audio_root)
for i in range(len(audios)):
    print(i)
    wavFile = os.path.join(audio_root, audios[i])
    complete_path = os.path.join(os.getcwd(), wavFile)
    # speech-to-text inference
    inference_time = 0.0
    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(wavFile, 1)
    texts = ""
    for i, segment in enumerate(segments):
        audio = np.frombuffer(segment, dtype=np.int16)
        output = wavTranscriber.stt(model_retval[0], audio, sample_rate)
        inference_time += output[1]
        texts += output[0] + " "
    print(texts, inference_time)
    audio_paths.append(complete_path.replace("\\", "/"))
    audio_texts.append(texts)

audio_paths = np.array(audio_paths)
audio_texts = np.array(audio_texts)
np.save("code/speech-to-text/audio_paths.npy", audio_paths)
np.save("code/speech-to-text/audio_texts.npy", audio_texts)
