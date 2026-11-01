"""
logistic_regression.py
Logistic Regression: a linear, probabilistic classifier.
Assumption: the log-odds of the positive class are a linear function of the features.
"""

from sklearn.linear_model import LogisticRegression
from tuning import validation_grid_search

RANDOM_STATE = 42


class LogisticRegressionModel:
    name = "Logistic Regression"

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],   # inverse regularisation strength
        "penalty": ["l1", "l2"],                # type of regularisation
    }

    def __init__(self):
        self.best_params = None
        self.tuning_results = None
        self.model = None

    @staticmethod
    def build(**params):
        return LogisticRegression(solver="liblinear", max_iter=5000,
                                  random_state=RANDOM_STATE, **params)

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
