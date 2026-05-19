"""Minimal GMR usage example."""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

try:
    from gmr import GMR
except ImportError:
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from gmr import GMR


def main() -> None:
    X, y = make_classification(
        n_samples=1200,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        weights=[0.90, 0.10],
        class_sep=0.8,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    baseline = RandomForestClassifier(class_weight="balanced", random_state=42)
    baseline.fit(X_train, y_train)
    baseline_score = average_precision_score(y_test, baseline.predict_proba(X_test)[:, 1])

    rectifier = GMR(random_state=42)
    X_gmr, y_gmr = rectifier.fit_resample(X_train, y_train)

    model = RandomForestClassifier(class_weight="balanced", random_state=42)
    model.fit(X_gmr, y_gmr)
    gmr_score = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Original training shape: {X_train.shape}, minority={np.sum(y_train == 1)}")
    print(f"GMR training shape:      {X_gmr.shape}, minority={np.sum(y_gmr == 1)}")
    print(f"Baseline AUPRC:         {baseline_score:.4f}")
    print(f"GMR AUPRC:              {gmr_score:.4f}")
    print(f"GMR audit:              {rectifier.audit_}")


if __name__ == "__main__":
    main()
