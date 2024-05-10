from pydub import AudioSegment
import math
import os
import librosa

def split(from_second, to_second, audio, out_path):

    start = from_second * 1000
    end = to_second * 1000
    split_audio = audio[start:end]
    split_audio.export(out_path, format="wav")
    
def multiple_split(directory_path, file_name, seconds_per_split):

    audio = AudioSegment.from_wav(directory_path + '/' + file_name + '.wav')

    duration_in_seconds = math.ceil(audio.duration_seconds)

    for i in range(0, duration_in_seconds, seconds_per_split):
        split(i, i + seconds_per_split, audio, f"{directory_path}/split/{file_name}_{i}.wav")