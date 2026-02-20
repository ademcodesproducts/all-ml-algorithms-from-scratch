import numpy as np

class StandardScaler:

    def fit(self, X):
        """
        fits column-wise (each feature)
        Only accepts Pandas Dataframes as X inputs
        """
        self.mean_, self.std_ = [], []

        for col in range(X.shape[1]):
            self.mean_.append(np.mean(X.iloc[:, col] ))
            self.std_.append(np.std(X.iloc[:, col]))
        
        return self
    
    def transform(self, X):
        X_scal = np.zeros((X.shape[0], X.shape[1]))

        # np.ndindex() generates all index tuples for an array of given shape
        for row, col in np.ndindex(X.shape):
            val_scal = (X.iloc[row, col] - self.mean_[col]) / self.std_[col] if self.std_[col] > 0 else 0
            X_scal[row, col] = val_scal

        return X_scal