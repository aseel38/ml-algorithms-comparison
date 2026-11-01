"""
evaluate.py
Test-set metrics and figures (confusion matrices, metric comparison chart).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

LABELS = ["Benign", "Malignant"]


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp),
        "F1-score": f1_score(y_true, y_pred),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }, cm


def plot_confusion_matrices(cms, names, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.2 * len(cms), 4))
    for ax, cm, name in zip(axes, cms, names):
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.sum() / 2 else "black")
        ax.set_xticks([0, 1], LABELS)
        ax.set_yticks([0, 1], LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric_comparison(df, path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    x = np.arange(len(metrics))
    width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (model, row) in enumerate(df.iterrows()):
        bars = ax.bar(x + (i - 1.5) * width, row[metrics], width,
                      label=model, color=colors[i])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{b.get_height():.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0.85, 1.025)
    ax.set_ylabel("Score (test set)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_knn_tuning(results, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for (w, m), grp in results.groupby(["weights", "metric"]):
        grp = grp.sort_values("n_neighbors")
        ax.plot(grp["n_neighbors"], grp["val_f1"], marker="o", label=f"{w}, {m}")
    ax.set_xlabel("k (n_neighbors)")
    ax.set_ylabel("Validation F1")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_lr_tuning(results, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for pen, grp in results.groupby("penalty"):
        grp = grp.sort_values("C")
        ax.plot(grp["C"], grp["val_f1"], marker="o", label=pen)
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Validation F1")
    ax.legend(title="penalty", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
