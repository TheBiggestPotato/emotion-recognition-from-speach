from pydub import AudioSegment
import math
import os

class WavFileProcessor():

    def __init__(self, folder, filename):

        self.folder = folder
        self.filename = filename
        self.filepath = os.path.join(folder, filename)
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):

        return self.audio.duration_seconds
    
    def split(self, from_second, to_second, filename):

        start = from_min * 1000
        end = to_min * 1000
        split_audio = self.audio[start:end]
        split_audio.export(os.path.join(self.folder, filename), format="wav")
        
    def multiple_split(self, seconds_per_split):

        duration_in_seconds = math.ceil(self.get_duration())

        for i in range(0, duration_in_seconds, seconds_per_split):
            split_file_name = f"{i}_{self.filename}"
            self.single_split(i, i + seconds_per_split, split_file_name)

            if i == duration_in_seconds - seconds_per_split:
                print("All splits completed successfully.")

    def stft(signal):

        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_size, win_length=self.frame_size)
        stft = stft / np.max(np.abs(stft))
        stft = np.abs(stft) ** 2
        stft = 20 * np.log10(stft + 1e-8)
        stft = np.clip(stft, -80, 0)
        stft = (stft / 40 ) + 1  
        stft = np.expand_dims(stft, axis=0) 
        
        return stft