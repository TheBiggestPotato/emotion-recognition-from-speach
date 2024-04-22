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

class Preprocessor:
    
    def __init__(self, config, mode : str = "train"):

        with open(config, 'r') as f:
            self.config = json.load(f)

        self.metadata_loader = MetadataLoader(config)
        self.metadata = self.metadata_loader.get_metadata()
        
        self.mode = mode

        self.train_set, self.test_set = self.split_dataset()

    def split_dataset(self):

        filenames_array = self.metadata["sentence_filenames"].to_numpy() 
        train_set, test_set = train_test_split(filenames_array, test_size = self.config["test_size"], random_state = self.config["random_state"])
        
        Utility.create_csv_from_array(test_set, self.config["path"] + self.config["test_set_filename"], ['Stimulus_Number', 'Filename'])

        return train_set, test_set


class DatasetLoader(Dataset):

    def __init__(self, metadata, config):
        
        with open(config, 'r') as f:
            self.config = json.load(f)

        self.metadata_loader = MetadataLoader(config)
        self.metadata = self.metadata_loader.get_metadata()


        self.frame_size = self.config["frame_size"]
        self.hop_size = self.config["hop"]
        self.sample_rate = self.config["sample_rate"]

    def __len__(self):
        return(len(self.metadata["sentence_filenames"]))

    def __getitem__(self):
        pass

if __name__ == "__main__":
    config = "config.json"
    preprocessor = Preprocessor(config, "train")
    preprocessor.split_dataset()
    print("xxxxxxxxxxxxxxxxxxxxx")
    print(len(preprocessor.metadata["sentence_filenames"]))