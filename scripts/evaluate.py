#!/usr/bin/env python3
"""AUROC with cluster-bootstrap 95% confidence intervals for one or more score files.

    python scripts/evaluate.py --labels data/labels/mimic_iv_echo_probe.csv \
        --seed-key mimic/patient_cluster/absolute --missing drop \
        --scores results/mimic_iv_echo/probe/*_test_scores.csv --out results/mimic_iv_echo/probe/auroc.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import load_labels  # noqa: E402
from eval.analysis.metrics import auroc_ci, cluster_draws, seed_from  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed-key", default=None, help="text key for the bootstrap seed (default: label file name)")
    ap.add_argument("--missing", choices=["fail", "drop"], default="fail")
    ap.add_argument("--tasks", nargs="+", default=None, help="tasks to run (default: all). Bootstrap draws still use every study of the label file")
    a = ap.parse_args()

    labels = load_labels(a.labels)
    test = pd.read_csv(a.labels, dtype=str)
    test = test[test.split == "TEST"] if "split" in test else test
    clusters = np.array(sorted(test.get("cluster_id", test.study_id).unique()))
    seed = seed_from(a.seed_key or Path(a.labels).name)
    draws = cluster_draws(len(clusters), a.replicates, seed)
    scores = pd.concat([pd.read_csv(f, dtype={"study_id": str}) for f in a.scores], ignore_index=True)
    if a.tasks:
        scores = scores[scores.task.isin(a.tasks)]

    rows = []
    for (method, task), s in scores.groupby(["method", "task"], sort=False):
        lab = labels[labels.task == task]
        if lab.empty:
            continue
        d = lab.merge(s[["study_id", "score"]], on="study_id", how="left")
        n_missing = int(d.score.isna().sum())
        if n_missing and a.missing == "fail":
            raise SystemExit(f"{method}/{task}: {n_missing} studies without a score (use --missing drop)")
        d = d.dropna(subset=["score"])
        if d.label.nunique() < 2:
            print(f"{method}/{task}: skipped, {len(d)} scored studies do not contain both classes")
            continue
        ci = auroc_ci(d.label.to_numpy(), d.score.to_numpy(), np.searchsorted(clusters, d.cluster_id), draws)
        rows.append(dict(method=method, task=task, n=len(d), n_positive=int(d.label.sum()), n_missing=n_missing, **ci))
    out = pd.DataFrame(rows)
    if out.empty:
        raise SystemExit("no task has scored studies of both classes")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    summary = out.groupby("method").agg(tasks=("task", "size"), mean_auroc=("auroc", "mean"))
    print(summary.to_string(float_format="%.3f"))
    print(f"seed {seed}, {a.replicates} replicates -> {a.out}")


if __name__ == "__main__":
    main()
