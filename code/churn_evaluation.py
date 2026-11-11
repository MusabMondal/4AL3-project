"""Model-independent experiment and binary classification metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
)
from sklearn.pipeline import Pipeline

from churn_data import FEATURES, TARGET
from churn_models import ProbabilisticClassifier

THRESHOLD = 0.5


def evaluate(labels: pd.Series, scores: np.ndarray, threshold: float = THRESHOLD) -> dict:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(labels) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected one finite probability in [0,1] per label.")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one.")
    predicted = scores > threshold
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, zero_division=0),
        "recall": recall_score(labels, predicted, zero_division=0),
        "f1": f1_score(labels, predicted, zero_division=0),
        "average_precision": average_precision_score(labels, scores),
        "roc_auc": roc_auc_score(labels, scores),
        "brier": brier_score_loss(labels, scores),
        "log_loss": log_loss(labels, scores, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


class ChurnExperiment:
    def __init__(self, preprocessor: Pipeline, classifier: ProbabilisticClassifier):
        self.pipeline = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])

    def fit(self, train: pd.DataFrame) -> None:
        self.pipeline.fit(train[FEATURES], train[TARGET])

    def score(self, customers: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(customers[FEATURES])[:, 1]
