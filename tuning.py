"""
tuning.py
Shared grid-search helper used by all four algorithm files.

For every combination in the grid:
  * fit on the TRAINING set and score F1 on the VALIDATION set  (primary criterion)
  * 5-fold stratified cross-validation F1 on the TRAINING set only (tie-breaker)

The validation set has only 86 samples, so several configurations can reach the
same validation score. The cross-validation score (which never touches the
validation or test sets) is used to choose between them. The test set is never used.
"""

import warnings
from itertools import product

import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*penalty is deprecated.*")


def validation_grid_search(build_model, param_grid, X_train, y_train, X_val, y_val):
    """
    build_model : function(**params) -> unfitted sklearn estimator
    param_grid  : dict {param_name: [values]}
    Returns (best_params, results_dataframe sorted best-first)
    """
    keys = list(param_grid.keys())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    records, configs = [], []
    for values in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        model = build_model(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        cv_f1 = cross_val_score(build_model(**params), X_train, y_train,
                                cv=cv, scoring="f1").mean()
        records.append({**params,
                        "val_f1": f1_score(y_val, pred),
                        "val_accuracy": accuracy_score(y_val, pred),
                        "cv_f1_train": cv_f1})
        configs.append(params)

    results = pd.DataFrame(records)
    results["_cfg"] = range(len(configs))
    results = results.sort_values(["val_f1", "cv_f1_train", "val_accuracy"],
                                  ascending=False, kind="stable")
    best = configs[int(results.iloc[0]["_cfg"])]
    results = results.drop(columns="_cfg").reset_index(drop=True)
    return best, results
