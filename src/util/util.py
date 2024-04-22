import numpy as np
import pandas as pd

class Utility:

    def create_csv_from_array(array, filename, columns):

        df = pd.DataFrame(array, columns=columns)
        df.to_csv(filename, index=False)
