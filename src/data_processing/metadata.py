import pandas as pd
import json

class MetadataLoader:

    def __init__(self, config):
        with open(config, 'r') as f:
            self.config = json.load(f)  
        
        self.load_metadata() 

    def load_metadata(self):
        
        csv_path = self.config['path']
        sentence_filenames_path = f"{csv_path}/SentenceFilenames.csv"
        finished_emo_responses_path = f"{csv_path}/finishedEmoResponses.csv"
        finished_responses_path = f"{csv_path}/finishedResponses.csv"
        finished_responses_with_repeats_path = f"{csv_path}/finishedResponsesWithRepeatWithPractice.csv"
        tabulated_votes_path = f"{csv_path}/processedResults/tabulatedVotes.csv"
        video_demographics_path = f"{csv_path}/VideoDemographics.csv"
        
        self.sentence_filenames = pd.read_csv(sentence_filenames_path)
        self.finished_emo_responses = pd.read_csv(finished_emo_responses_path)
        self.finished_responses = pd.read_csv(finished_responses_path)
        self.finished_responses_with_repeats = pd.read_csv(finished_responses_with_repeats_path)
        self.tabulated_votes = pd.read_csv(tabulated_votes_path)
        self.video_demographics = pd.read_csv(video_demographics_path)

    def get_metadata(self):

        return {
            "sentence_filenames": self.sentence_filenames,
            "finished_emo_responses": self.finished_emo_responses,
            "finished_responses": self.finished_responses,
            "finished_responses_with_repeats": self.finished_responses_with_repeats,
            "tabulated_votes": self.tabulated_votes,
            "video_demographics": self.video_demographics
        }