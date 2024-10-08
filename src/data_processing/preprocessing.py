import json
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from metadata import MetadataLoader
import sys
sys.path.insert(1, "src/util")
from util import Utility
import pandas as pd
import numpy as np
import librosa
from wav import multiple_split
import torch
import os
from sklearn.preprocessing import StandardScaler
from scipy.io import wavfile
import matplotlib.pyplot as plt

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

        self.max_stft_length = self.calculate_max_stft_length()

        self.directory_path = f"{self.config['path']}audio"

        # self.split_all_wav_files(self.directory_path, 1)
        # self.pad_wav_files(self.directory_path + '/split')


    def pad_wav_files(self, folder_path, target_duration=1):
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(folder_path, filename)
                
                sr, audio_data = wavfile.read(file_path)

                length_in_seconds = len(audio_data) / sr

                if length_in_seconds < target_duration:
                    target_samples = int(target_duration * sr)

                    samples_to_add = target_samples - len(audio_data)

                    padded_audio_data = np.pad(audio_data, (0, samples_to_add), mode='constant')

                    wavfile.write(file_path, sr, padded_audio_data)

    def split_dataset(self):

        filenames_array = self.metadata["sentence_filenames"].to_numpy() 
        train_set, test_set = train_test_split(filenames_array, test_size = self.config["test_size"], random_state = self.config["random_state"])
        
        Utility.create_csv_from_array(test_set, self.config["path"] + self.config["test_set_filename"], ['Stimulus_Number', 'Filename'])
        Utility.create_csv_from_array(train_set, self.config["path"] + self.config["train_set_filename"], ['Stimulus_Number', 'Filename'])

        return train_set, test_set

    def get_stft(self, signal):

        n_fft = min(self.frame_size, len(signal))
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=self.hop_size, win_length=self.frame_size)
        epsilon = 1e-8
        max_abs_stft = np.max(np.abs(stft))
        stft = stft / (max_abs_stft + epsilon)
        stft = np.abs(stft) ** 2
        stft = 20 * np.log10(stft + 1e-8)
        stft = np.clip(stft, -80, 0)
        stft = (stft / 40 ) + 1  
        stft = np.expand_dims(stft, axis=0) 
        
        return stft

    def get_emotion_label_and_level(self, filename):

        label_row = self.metadata["finished_responses"][self.metadata["finished_responses"]["clipName"] == filename]

        emotion = None
        emotion_level = None

        if not label_row.empty:

            emotion = label_row.iloc[0]["respEmo"]
            emotion_level = label_row.iloc[0]["respLevel"]
        
        emotion = int(Utility.emotion_label_to_number(emotion))
            
        return emotion, int(emotion_level)

    def __len__(self):
        return(len(self.train_set) if self.mode == "train" else len(self.test_set))

    def calculate_max_stft_length(self):
        max_length = 0
        for filename in self.metadata["sentence_filenames"]["Filename"]:
            file_path = f"{self.config['path']}/audio/{filename}.wav"
            signal, _ = librosa.load(file_path, sr=self.sample_rate)
            stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_size, win_length=self.frame_size)
            max_length = max(max_length, stft.shape[1]) 
        return max_length

    def pad_or_truncate_stft(self, stft):
        current_length = stft.shape[2]
        if current_length < self.max_stft_length:
            padding_length = self.max_stft_length - current_length
            stft = np.pad(stft, ((0, 0), (0, 0), (0, padding_length)), mode='constant', constant_values=0)
        elif current_length > self.max_stft_length:
            stft = stft[:, :, :self.max_stft_length]
        
        return stft

    def split_all_wav_files(self, directory_path, seconds_per_split):
        files = os.listdir(self.directory_path)
        
        wav_files = [f for f in files if f.endswith('.wav')]

        for file_name in wav_files:
            base_name = os.path.splitext(file_name)[0]

            multiple_split(self.directory_path, base_name, seconds_per_split)

    def normalize_data(self, stft_array):
        scaler = StandardScaler()
        stft_shape = stft_array.shape
        reshaped_stft = stft_array.reshape(stft_shape[0], -1)
        reshaped_stft = scaler.fit_transform(reshaped_stft)

        stft_array = reshaped_stft.reshape(stft_shape)
        return stft_array


    def __getitem__(self, index):
        
        if self.mode == "train":
            row = self.train_set[index]
        else:
            row = self.test_set[index]

        filename = row[1]

        all_files = os.listdir(self.directory_path + "/split")

        file_path = f"{self.config['path']}audio/{filename}.wav"

        stft_array = []
        
        for audio_file in all_files:
            if audio_file.startswith(filename):
                signal, sr = librosa.load(f"{self.config['path']}audio/split/{audio_file}", sr=self.sample_rate)
                stft = self.get_stft(signal)
                stft = self.pad_or_truncate_stft(stft)
                stft = np.squeeze(stft)
                stft_array.append(stft)

        emotion, emotion_level = self.get_emotion_label_and_level(filename)

        stft_array = np.stack(stft_array, axis=0)


        stft_array = self.normalize_data(stft_array)
        
        return {
            "stft" : stft_array,
            "emotion": emotion,
            "emotion_level" : emotion_level
        }
    
    def viz(self, audio_file_path):
        # Load the audio file
        audio, sr = librosa.load(audio_file_path, sr=None)

        # Compute the STFT
        n_fft = 512  # Number of FFT points
        hop_length = 256  # Number of samples between successive frames
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

        # Compute the magnitude and phase of the STFT
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Create two plots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot the spectrogram (magnitude)
        img1 = librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max), sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax1)
        fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        ax1.set(title='Spectrogram (Magnitude)')
        
        # Save the spectrogram
        plt.savefig('spectrogram_magnitude.png')

        # Plot the phaseogram (phase)
        img2 = librosa.display.specshow(phase, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax2)
        fig.colorbar(img2, ax=ax2, format='%+2.2f rad')
        ax2.set(title='Phaseogram (Phase)')
        
        # Save the phaseogram
        plt.savefig('phaseogram_phase.png')

        # Show the plots
        plt.show()


if __name__ == "__main__":

    config_file_path = 'config.json'

    preprocessor = Preprocessor(config_file_path)

    item_index_0 = preprocessor.__getitem__(0)

    preprocessor.viz("data/audio/1001_DFA_ANG_XX.wav")

    print("STFT shape:", item_index_0["stft"].shape)
    print(item_index_0["emotion"])
    print(item_index_0["emotion_level"])