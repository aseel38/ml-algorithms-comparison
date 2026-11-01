"""
main.py
Runs the full pipeline:
  1. load + split (70/15/15) + standardise
  2. tune each model on the validation set
  3. refit with the best hyperparameters on the training set
  4. evaluate ONCE on the untouched test set
All tables go to results/, all figures to figures/.

Run:  python main.py
"""

import json
import os

import pandas as pd

from data_prep import DataPreparer
from logistic_regression import LogisticRegressionModel
from knn import KNNModel
from svm import SVMModel
from random_forest import RandomForestModel
from evaluate import (compute_metrics, plot_confusion_matrices,
                      plot_metric_comparison, plot_knn_tuning, plot_lr_tuning)

os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


def main():
    prep = DataPreparer()
    summary, shape = prep.describe()
    print(f"Dataset shape: {shape}")
    print(summary.to_string(index=False))
    summary.to_csv("results/split_summary.csv", index=False)

    X_train, X_val, X_test, y_train, y_val, y_test = prep.prepare()

    models = [LogisticRegressionModel(), KNNModel(), SVMModel(), RandomForestModel()]

    # ---- Phase 1: tuning on the validation set (test set not used) ----
    best_params = {}
    for m in models:
        params = m.tune(X_train, y_train, X_val, y_val)
        best_params[m.name] = params
        n = len(m.tuning_results)
        top = m.tuning_results.iloc[0]
        print(f"\n{m.name}: {n} configurations tried")
        print(f"  best params: {params}  | val F1 = {top.val_f1:.4f}, cv F1 = {top.cv_f1_train:.4f}, "
              f"val acc = {top.val_accuracy:.4f}")
        fname = m.name.lower().replace(" ", "_").replace("-", "_")
        m.tuning_results.to_csv(f"results/tuning_{fname}.csv", index=False)

    with open("results/best_params.json", "w") as f:
        json.dump(best_params, f, indent=2, default=str)

    # ---- Phase 2: final fit + single evaluation on the test set ----
    rows, cms = [], []
    for m in models:
        m.fit(X_train, y_train)
        metrics, cm = compute_metrics(y_test, m.predict(X_test))
        rows.append({"Model": m.name, **metrics})
        cms.append(cm)

    results = pd.DataFrame(rows).set_index("Model")
    print("\nTEST SET RESULTS")
    print(results.round(4).to_string())
    results.round(4).to_csv("results/test_metrics.csv")

    plot_confusion_matrices(cms, [m.name for m in models],
                            "figures/confusion_matrices.png")
    plot_metric_comparison(results, "figures/metric_comparison.png")
    plot_lr_tuning(models[0].tuning_results, "figures/tuning_logreg.png")
    plot_knn_tuning(models[1].tuning_results, "figures/tuning_knn.png")


if __name__ == "__main__":
    main()
