"""
data_prep.py
Loads the Breast Cancer Wisconsin (Diagnostic) dataset, splits it into
train / validation / test (70/15/15, stratified) and standardises the features.

The scaler is fitted on the TRAINING set only, then applied to the validation
and test sets, so no information from those sets leaks into training.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class DataPreparer:
    """Handles loading, splitting and scaling of the dataset."""

    def __init__(self, test_size=0.15, val_size=0.15, random_state=RANDOM_STATE):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = None

    def load(self):
        data = load_breast_cancer()
        self.feature_names = list(data.feature_names)
        # sklearn encodes 0 = malignant, 1 = benign. We flip it so that the
        # clinically important class (malignant) is the positive class = 1.
        self.target_names = ["benign", "malignant"]
        X = data.data
        y = 1 - data.target
        return X, y

    def split(self, X, y):
        # Step 1: carve out the test set (15%) and lock it away.
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y,
            random_state=self.random_state)
        # Step 2: split the remaining 85% into train (70%) and validation (15%).
        val_fraction = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_fraction, stratify=y_temp,
            random_state=self.random_state)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale(self, X_train, X_val, X_test):
        X_train_s = self.scaler.fit_transform(X_train)   # fit on train ONLY
        X_val_s = self.scaler.transform(X_val)
        X_test_s = self.scaler.transform(X_test)
        return X_train_s, X_val_s, X_test_s

    def prepare(self):
        X, y = self.load()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y)
        X_train, X_val, X_test = self.scale(X_train, X_val, X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    def describe(self):
        """Summary statistics used in the report's introduction."""
        X, y = self.load()
        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y)
        rows = []
        for name, labels in [("Train", y_train), ("Validation", y_val),
                             ("Test", y_test), ("Total", y)]:
            rows.append({"Split": name, "Samples": len(labels),
                         "Benign (0)": int((labels == 0).sum()),
                         "Malignant (1)": int((labels == 1).sum()),
                         "Malignant %": round(100 * labels.mean(), 1)})
        return pd.DataFrame(rows), X.shape
