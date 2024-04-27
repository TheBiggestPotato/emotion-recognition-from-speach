import numpy as np
import pandas as pd
import os

class Utility:

    def create_csv_from_array(array, filename, columns):

        df = pd.DataFrame(array, columns=columns)
        df.to_csv(filename, index=False)

    def emotion_label_to_number(label):
        
        label_to_number = {
            'A': 0,
            'D': 1,
            'F': 2,
            'H': 3,
            'N': 4,
            'S': 5,
        }
        
        return label_to_number.get(label, None)

    def number_to_emotion_label(number):

        number_to_label = {
            0: 'A',
            1: 'D',
            2: 'F',
            3: 'H',
            4: 'N',
            5: 'S',
        }

        return number_to_label.get(number, None)
