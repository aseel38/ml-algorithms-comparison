# Comparative Analysis of Machine Learning Algorithms

Breast cancer diagnosis (benign vs. malignant) on the **Breast Cancer Wisconsin (Diagnostic)** dataset, comparing four tuned classifiers:

| Model | Test Accuracy | Test F1 | Missed cancers (FN) |
|---|---|---|---|
| Logistic Regression | 1.000 | 1.000 | 0 |
| Support Vector Machine | 0.977 | 0.969 | 1 |
| Random Forest | 0.965 | 0.951 | 3 |
| k-Nearest Neighbors | 0.919 | 0.889 | 4 |

## Method
- **Split:** stratified 70 / 15 / 15 (train / validation / test), `random_state=42`. The test set is used only once, at the end.
- **Scaling:** `StandardScaler` fitted on the training set only.
- **Tuning:** grid search; each configuration is trained on the training set and scored by F1 on the validation set. Ties are broken by 5-fold cross-validation F1 on the training set only.
- **Evaluation:** confusion matrix, accuracy, precision, recall, specificity, F1 (malignant = positive class).

## Project structure
```
data_prep.py            # load, split, scale (DataPreparer class)
tuning.py               # shared validation grid search
logistic_regression.py  # LogisticRegressionModel
knn.py                  # KNNModel
svm.py                  # SVMModel
random_forest.py        # RandomForestModel
evaluate.py             # metrics and figures
main.py                 # runs everything
results/                # CSV/JSON outputs
figures/                # PNG figures
```
Every model class has the same interface: `tune()`, `fit()`, `predict()`.

## How to run
```bash
pip install -r requirements.txt
python main.py
```
Runs in about 1–2 minutes and regenerates everything in `results/` and `figures/`.

## Authors
Aseel ([@aseel38](https://github.com/aseel38)) & Moayad
