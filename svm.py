"""
svm.py
Support Vector Machine: finds the maximum-margin separating boundary.
With the RBF kernel it can learn non-linear boundaries.
"""

from sklearn.svm import SVC
from tuning import validation_grid_search

RANDOM_STATE = 42


class SVMModel:
    name = "Support Vector Machine"

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", 0.001, 0.01, 0.1],   # only affects the rbf kernel
    }

    def __init__(self):
        self.best_params = None
        self.tuning_results = None
        self.model = None

    @staticmethod
    def build(**params):
        return SVC(random_state=RANDOM_STATE, **params)

    def tune(self, X_train, y_train, X_val, y_val):
        self.best_params, self.tuning_results = validation_grid_search(
            self.build, self.param_grid, X_train, y_train, X_val, y_val)
        return self.best_params

    def fit(self, X_train, y_train):
        self.model = self.build(**self.best_params)
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)
