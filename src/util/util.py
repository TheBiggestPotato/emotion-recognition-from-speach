import numpy as np
import pandas as pd

class Utility:

    def create_csv_from_array(array, filename, columns):

        df = pd.DataFrame(array, columns=columns)
        df.to_csv(filename, index=False)

    def emotion_label_to_number(label):
        
        label_to_number = {
            'A': 1,
            'D': 2,
            'F': 3,
            'H': 4,
            'N': 5,
            'S': 6,
        }
        
        return label_to_number.get(label, None)

    def number_to_emotion_label(number):

        number_to_label = {
            1: 'A',
            2: 'D',
            3: 'F',
            4: 'H',
            5: 'N',
            6: 'S',
        }

        return number_to_label.get(number, None)
