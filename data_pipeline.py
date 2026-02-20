import pandas as pd
import numpy as np
from preprocessing.scaler import StandardScaler

class DataPipeline:
    def __init__(self, X, y):
        self.scaler = StandardScaler()

    def read_csv(data):
        return pd.read_csv(data)

    def standard_scaler(X, ):
        s = np.std(X)
        z_score = (X - np.mean(X)) / s

        print('nigeria')

    def normalizer():
        """
        Min-Max scaling
        """
        print('uganda')

    def train_test_split():
        print('namibia')
