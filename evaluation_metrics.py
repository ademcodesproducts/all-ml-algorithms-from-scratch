import numpy as np


class EvaluationMetrics:
    """Combined classification and regression metrics in one class to demonstrate OOP understanding.
    In production, these would typically be standalone functions like sklearn.metrics does it.
    """

    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self._get_confusion_matrix_values()

    def _get_confusion_matrix_values(self):
        self.tp = sum(
            1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 1 and yp == 1
        )
        self.tn = sum(
            1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 0 and yp == 0
        )
        self.fp = sum(
            1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 0 and yp == 1
        )
        self.fn = sum(
            1 for yt, yp in zip(self.y_true, self.y_pred) if yt == 1 and yp == 0
        )

    # Classification Metrics

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def specificity(self):
        return self.tn / (self.tn + self.fp)

    def f1_score(self):
        if self.precision() + self.recall() == 0:
            return 0
        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    # Regression Metrics

    def mae(self):
        return 1 / len(self.y_true) * sum(np.abs(self.y_true - self.y_pred))

    def rmse(self):
        return np.sqrt(
            1 / len(self.y_true) * sum(np.power(self.y_true - self.y_pred, 2))
        )

    def mape(self):
        if np.any(self.y_true == 0):
            return 0
        return (
            100
            / len(self.y_true)
            * sum(np.abs((self.y_true - self.y_pred) / self.y_true))
        )

    def r_squared(self):
        ss_res = np.sum(np.power(self.y_true - self.y_pred, 2))
        ss_tot = np.sum(np.power(self.y_true - np.mean(self.y_true), 2))
        return 1 - (ss_res / ss_tot)
