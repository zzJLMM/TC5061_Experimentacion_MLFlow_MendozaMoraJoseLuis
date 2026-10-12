"""Metricas de prueba y matriz de confusion del clasificador."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

SPAM_LABEL = 1


def compute_metrics(
    labels: pd.Series,
    predictions: pd.Series,
) -> dict[str, float]:
    """Calcula las metricas de spam y el piso de la clase mayoritaria."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_spam": float(
            precision_score(labels, predictions, pos_label=SPAM_LABEL)
        ),
        "recall_spam": float(
            recall_score(labels, predictions, pos_label=SPAM_LABEL)
        ),
        "f1_spam": float(f1_score(labels, predictions, pos_label=SPAM_LABEL)),
        "accuracy_majority_ham": float((labels == 0).mean()),
    }


def save_confusion_matrix(
    labels: pd.Series,
    predictions: pd.Series,
    figure_path: Path,
) -> Path:
    """Guarda la matriz de confusion como imagen para MLflow."""
    counts = confusion_matrix(labels, predictions)
    figure, axis = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        counts,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
        ax=axis,
    )
    axis.set_xlabel("Prediccion")
    axis.set_ylabel("Real")
    axis.set_title("Matriz de confusion en prueba")
    figure.tight_layout()

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)
    return figure_path
