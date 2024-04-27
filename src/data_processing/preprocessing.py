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

    def __init__(self, config, mode : str = "train"):
        
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
        
        emotion = int(Utility.emotion_label_to_number(emotion))
            
        return emotion, emotion_level

    def __len__(self):
        return(len(self.train_set) if self.mode == "train" else len(self.test_set))

    def __getitem__(self, index):
        
        if self.mode == "train":
            row = self.train_set[index]
        else:
            row = self.test_set[index]

        filename = row[1]
        
        file_path = f"{self.config['path']}/audio/{filename}.wav"

        signal, sr = librosa.load(file_path, sr=self.sample_rate)

        emotion, emotion_level = self.get_emotion_label_and_level(filename)
        
        stft = stft(signal)
        
        print(stft.shape)

        return {
            "stft" : stft,
            "emotion": emotion,
            "emotion_level" : emotion_level
        }

if __name__ == "__main__":

    config_file_path = 'config.json'

    preprocessor = Preprocessor(config_file_path)

    item_index_0 = preprocessor.__getitem__(0)

    print("STFT shape:", item_index_0["stft"].shape)
    print(item_index_0["emotion"])
    print(item_index_0["emotion_level"])