#!/usr/bin/env python3
"""Collect the paired comparisons of several cohorts into long tables, and pool tasks across cohorts.

    python scripts/collect_results.py --root results --cohorts mimic_iv_echo cardiobench \
        --pair echoalign:echoprime --out results/collected
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.metrics import interval  # noqa: E402

KINDS = {"compare_test_scores/paired": "probe", "compare_regression_predictions/paired": "probe",
         "compare_ordinal_predictions/paired": "probe", "zeroshot/paired": "zeroshot"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--rename", action="append", default=[], help="OLD=NEW")
    ap.add_argument("--pair", action="append", default=[], help="A:B (A minus B)")
    ap.add_argument("--exclude-tasks", nargs="+", default=[], help="tasks left out of every table")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    renames = [r.split("=", 1) for r in a.rename]

    def name(m):
        for old, new in renames:
            if m == old:
                return new
        return m

    per, draws = [], {}
    for cohort in a.cohorts:
        for folder, evaluation in KINDS.items():
            d = Path(a.root) / cohort / folder
            if not (d / "per_task.csv").exists():
                print(f"missing {d}")
                continue
            t = pd.read_csv(d / "per_task.csv")
            t = t[~t.task.isin(a.exclude_tasks)]
            t["method"] = t.method.map(name)
            per.append(t.assign(cohort=cohort, evaluation=evaluation))
            z = np.load(d / "draws.npz")
            for k in z.files:
                m, task = k.split("|", 1)
                if task not in a.exclude_tasks:
                    metric = t[t.task == task].metric.iloc[0]
                    if (cohort, evaluation, metric, name(m), task) in draws:
                        raise SystemExit(f"--rename maps two methods to {name(m)}")
                    draws[(cohort, evaluation, metric, name(m), task)] = z[k]
    per = pd.concat(per, ignore_index=True)
    first = ["cohort", "evaluation", "mode", "metric", "method", "task"]
    per = per[first + [c for c in per.columns if c not in first]]

    paired, pooled = [], []
    groups = [("all", a.cohorts)] + [(c, [c]) for c in a.cohorts]
    for (evaluation, metric), g in per.groupby(["evaluation", "metric"]):
        for scope, cohorts in groups:
            sub = g[g.cohort.isin(cohorts)]
            keys = lambda m: {(r.cohort, r.task) for r in sub[sub.method == m].itertuples()}
            for m in sorted(set(sub.method)):
                ks = sorted(keys(m))
                v = np.mean([draws[(c, evaluation, metric, m, t)] for c, t in ks], 0)
                lo, hi, valid = interval(v)
                pooled.append(dict(scope=scope, evaluation=evaluation, metric=metric, comparison=m, tasks=len(ks),
                                   point=float(sub[sub.method == m].point.mean()), lower=lo, upper=hi, valid_replicates=valid))
            for pair in a.pair:
                x, y = pair.split(":")
                ks = sorted(keys(x) & keys(y))
                if not ks:
                    continue
                px = sub[sub.method == x].set_index(["cohort", "task"]).point
                py = sub[sub.method == y].set_index(["cohort", "task"]).point
                diff = [draws[(c, evaluation, metric, x, t)] - draws[(c, evaluation, metric, y, t)] for c, t in ks]
                lo, hi, valid = interval(np.mean(diff, 0))
                better = sum((px[k] < py[k]) if metric == "RMSE" else (px[k] > py[k]) for k in ks)
                pooled.append(dict(scope=scope, evaluation=evaluation, metric=metric, comparison=f"{x} minus {y}",
                                   tasks=len(ks), point=float(np.mean([px[k] - py[k] for k in ks])), lower=lo, upper=hi,
                                   valid_replicates=valid, tasks_a_better=int(better)))
                if scope == "all":
                    for (c, t), dv in zip(ks, diff):
                        lo, hi, valid = interval(dv)
                        paired.append(dict(cohort=c, evaluation=evaluation, metric=metric, a=x, b=y, task=t,
                                           difference=px[(c, t)] - py[(c, t)], lower=lo, upper=hi, valid_replicates=valid))

    pooled = [p for p in pooled if p["metric"] != "RMSE"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_task.csv", index=False, float_format="%.17g")
    pd.DataFrame(paired).to_csv(out / "paired_per_task.csv", index=False, float_format="%.17g")
    pd.DataFrame(pooled).to_csv(out / "pooled.csv", index=False, float_format="%.17g")
    print(pd.DataFrame(pooled).query("scope == 'all'").to_string(index=False))


if __name__ == "__main__":
    main()
