"""
knn.py
k-Nearest Neighbors: a non-parametric, instance-based classifier.
Assumption: samples that are close in feature space share the same label.
Very sensitive to feature scale, which is why standardisation matters here.
"""

from sklearn.neighbors import KNeighborsClassifier
from tuning import validation_grid_search


class KNNModel:
    name = "k-Nearest Neighbors"

    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 25],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    def __init__(self):
        self.best_params = None
        self.tuning_results = None
        self.model = None

    @staticmethod
    def build(**params):
        return KNeighborsClassifier(**params)

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
