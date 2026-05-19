# Geometric Manifold Rectification

Geometric Manifold Rectification (GMR) is a model-agnostic preprocessing method for imbalanced tabular classification. It audits local training-set geometry and only intervenes when the data show enough overlap, boundary conflict, sparsity, and imbalance pressure to justify a small correction.

The public API is intentionally compact:

```python
from gmr import GMR

rectifier = GMR(random_state=42)
X_resampled, y_resampled = rectifier.fit_resample(X_train, y_train)
print(rectifier.audit_)
```

GMR is designed to sit before any downstream binary classifier. It can return the original training split unchanged, prune a small number of risky majority boundary samples, and add bounded same-class synthetic samples in geometrically safer regions.

## Repository Name

Recommended name: `gmr-imbalanced-learning`

Short description:

> Geometric preprocessing for imbalanced tabular learning with a compact Python implementation and reproducible benchmark artifacts.

Suggested topics: `imbalanced-learning`, `tabular-data`, `resampling`, `geometric-learning`, `scikit-learn`, `machine-learning`, `reproducibility`.

## Contents

```text
gmr-imbalanced-learning/
|-- code/
|   |-- gmr/
|   |   |-- rectifier.py        # Main GMR implementation
|   |   `-- __init__.py         # Public API: GMR, GeometricManifoldRectifier
|   `-- examples/
|       `-- quick_start.py      # Minimal runnable example
|-- data/
|   |-- benchmark_results_25datasets/
|   `-- contemporary_baselines_25datasets/
|-- DATA_CARD.md
|-- CITATION.cff
|-- LICENSE
|-- requirements.txt
`-- setup.py
```

The manuscript source is not part of this repository. The included CSV files are derived result artifacts for checking reported trends and reproducing summary analyses without redistributing raw third-party datasets.

## Installation

```bash
git clone https://github.com/wangxb96/gmr-imbalanced-learning.git
cd gmr-imbalanced-learning
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```bash
python code/examples/quick_start.py
```

Or use the console entry point after installation:

```bash
gmr-quickstart
```

## Basic Usage

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from gmr import GMR

gmr = GMR(random_state=42)
X_gmr, y_gmr = gmr.fit_resample(X_train, y_train)

clf = RandomForestClassifier(class_weight="balanced", random_state=42)
clf.fit(X_gmr, y_gmr)
score = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])

print(score)
print(gmr.audit_)
```

## Data Boundary

This repository includes derived benchmark CSVs only. It does not include raw downloaded datasets, server working directories, experiment logs, manuscript files, or private scratch artifacts. See `DATA_CARD.md` for the data inventory.

## Citation

If this code or the derived artifacts are useful in your work, please cite the repository using the metadata in `CITATION.cff`.
