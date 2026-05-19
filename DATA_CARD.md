# Data Card

This repository includes derived CSV artifacts for auditing GMR benchmark results. It does not redistribute raw third-party datasets.

## Included Files

`data/benchmark_results_25datasets/` contains matched benchmark outputs over 25 public imbalanced tabular datasets, 7 classifiers, and 10 random seeds. The primary metric is AUPRC.

`data/contemporary_baselines_25datasets/` contains derived summaries for contemporary resampling baselines aligned to the same dataset accounting.

## Not Included

The release intentionally excludes raw downloaded datasets, server-side experiment shards, local reruns, migration logs, LaTeX sources, generated manuscript PDFs, and temporary build files.

## Use

The CSV files are intended for result inspection, independent plotting, and aggregate checks. Users who need a full rerun should obtain raw datasets from their original sources and follow their licenses.
