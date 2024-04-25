import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from metadata import MetadataLoader
import sys
sys.path.insert(1, "src/util")
from util import Utility
import pandas as pd
import numpy as np
import librosa

class Preprocessor(Dataset):

    def __init__(self, metadata, config, mode : str = "train"):
        
        with open(config, 'r') as f:
            self.config = json.load(f)

        self.metadata_loader = MetadataLoader(config)
        self.metadata = self.metadata_loader.get_metadata()

        self.mode = mode

        self.frame_size = self.config["frame_size"]
        self.hop_size = self.config["hop"]
        self.sample_rate = self.config["sample_rate"]

        self.train_set, self.test_set = self.split_dataset()

    def split_dataset(self):

        filenames_array = self.metadata["sentence_filenames"].to_numpy() 
        train_set, test_set = train_test_split(filenames_array, test_size = self.config["test_size"], random_state = self.config["random_state"])
        
        Utility.create_csv_from_array(test_set, self.config["path"] + self.config["test_set_filename"], ['Stimulus_Number', 'Filename'])
        Utility.create_csv_from_array(train_set, self.config["path"] + self.config["train_set_filename"], ['Stimulus_Number', 'Filename'])

        return train_set, test_set

    def get_emotion_label_and_level(self, filename):

        label_row = self.metadata["finished_responses"][self.metadata["finished_responses"]["clipName"] == filename]

        emotion = None
        emotion_level = None

        if not label_row.empty:

            emotion = label_row.iloc[0]["respEmo"]
            emotion_level = label_row.iloc[0]["respLevel"]
            
        return emotion, emotion_level

    def __len__(self):
        return(len(self.metadata["sentence_filenames"]))

    def __getitem__(self, index):

        if self.mode == "train":
            row = self.train_set[index]
        else:
            row = self.test_set[index]

        filename = row[1]
        

        file_path = f"{self.config['path']}/audio/{filename}.wav"
        signal, sr = librosa.load(file_path, sr=self.sample_rate)
        
        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_size, win_length=self.frame_size)
        stft = np.abs(stft) ** 2 
        stft = librosa.amplitude_to_db(stft, ref=np.max)

        stft = np.clip(stft, -80, 0)
        stft = np.expand_dims(stft, axis=0)
        
        amplitude = np.abs(signal)
        
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_size)
        
        loudness = librosa.power_to_db(np.square(amplitude), ref=np.max)
        
        envelope = librosa.onset.onset_strength(y=signal, sr=sr)
        
        emotion_label, emotion_level = self.get_emotion_label_and_level(filename)
        
        features = {
            'stft': stft,
            'amplitude': amplitude,
            'frequency': freqs,
            'loudness': loudness,
            'envelope': envelope
        }
        
        return features, (emotion_label, emotion_level)

if __name__ == "__main__":

    config_file_path = 'config.json'
    metadata_object = MetadataLoader(config_file_path)

    preprocessor = Preprocessor(metadata_object, config_file_path)

    item_index_0 = preprocessor.__getitem__(0)

    print(len(item_index_0[0]["amplitude"]))
    print("STFT shape:", item_index_0[0]["stft"].shape)
    print("Emotion label and level:", item_index_0[1])