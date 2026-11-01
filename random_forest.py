"""
random_forest.py
Random Forest: an ensemble of decision trees trained on bootstrap samples
with random feature subsets; predictions are made by majority vote.
Makes no linearity assumption and is insensitive to feature scale.
"""

from sklearn.ensemble import RandomForestClassifier
from tuning import validation_grid_search

RANDOM_STATE = 42


class RandomForestModel:
    name = "Random Forest"

    param_grid = {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [None, 4, 8, 12],
        "min_samples_split": [2, 5, 10],
    }

    def __init__(self):
        self.best_params = None
        self.tuning_results = None
        self.model = None

    @staticmethod
    def build(**params):
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)

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
