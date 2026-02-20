import numpy as np

class Normalizer:
    def __init__(self, norm='l2'):
        self.norm = norm

    def fit(self, X):
        """
        sklearn's fit method does nothing
        """
        return self

    def transform(self, X):
        X_norm = np.zeros((X.shape[0], X.shape[1]))

        #TODO: Vectorized approach instead of np.ndindex() loops since slow
        if self.norm == 'l2':
            for row, col in np.ndindex(X.shape):
                v_norm = np.sqrt(np.sum(np.power(X.iloc[row, :], 2)))
                val_norm = X.iloc[row, col] / v_norm if v_norm > 0 else 0
                X_norm[row, col] = val_norm

        elif self.norm == 'max':
            for row, col in np.ndindex(X.shape):
                v_norm = max(np.abs(X.iloc[row, :]))
                val_norm = X.iloc[row, col] / v_norm if v_norm > 0 else 0
                X_norm[row, col] = val_norm

        return X_norm