#!/usr/bin/env python3
"""AUROC, RMSE or QWK per task with 95% intervals, and paired differences between methods, on one cohort.

    C=results/mimic_iv_echo/compare_test_scores
    python scripts/compare_paired.py --labels $C/paired_labels.csv --universe $C/universe.csv \
        --seed-key mimic/patient_cluster/absolute --save-draws --method echoalign=$C/scores_echoalign.csv \
        --method echoprime=$C/scores_echoprime.csv --pair echoalign:echoprime --out $C/paired
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.metrics import interval, metric_draws, point, cluster_draws, seed_from  # noqa: E402


def read_method(path):
    d = pd.read_csv(path, dtype={"study_id": str, "task": str}, float_precision="round_trip")
    col = next(c for c in ("y_pred", "prediction", "score") if c in d)
    return d[["study_id", "task", col]].rename(columns={col: "score"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--method", action="append", required=True, help="NAME=CSV")
    ap.add_argument("--pair", action="append", default=[], help="A:B (A minus B)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seed-key", default=None)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--mean-tasks", nargs="+", default=None, help="classification tasks of the summary mean (default: all shared)")
    ap.add_argument("--save-draws", action="store_true",
                    help="also write draws.npz: each method and task's metric per replicate, to pool tasks across "
                         "cohorts (scripts/collect_results.py)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if (a.seed is None) == (a.seed_key is None):
        ap.error("give exactly one of --seed and --seed-key")
    seed = a.seed if a.seed is not None else seed_from(a.seed_key)

    lab = pd.read_csv(a.labels, dtype={"study_id": str, "task": str, "split": str, "cluster_id": str})
    grades = lab.groupby("task").label.max().astype(int) + 1
    lab = lab[lab.split == "TEST"].copy()
    if "cluster_id" not in lab:
        lab["cluster_id"] = lab["study_id"]
    uni = pd.read_csv(a.universe, dtype=str)
    clusters = uni["cluster_id" if "cluster_id" in uni else "study_id"].to_numpy()
    col = pd.Series(np.arange(len(clusters)), index=clusters)
    if not lab.cluster_id.isin(col.index).all():
        raise SystemExit("some TEST clusters are not in --universe")
    draws = cluster_draws(len(clusters), a.replicates, seed)

    methods = dict(m.split("=", 1) for m in a.method)
    boots, rows = {}, []
    for name, path in methods.items():
        d = read_method(path)
        d = d[d.groupby("task").score.transform(lambda v: v.notna().any())]
        for task, s in d.groupby("task", sort=False):
            t = lab[lab.task == task]
            if t.empty:
                raise SystemExit(f"{name}: task {task} has no TEST labels")
            mode = t["mode"].fillna("classification").iloc[0] if "mode" in t else ("classification" if t.label.isin([0, 1]).all() else "regression")
            metric = dict(classification="AUROC", ordinal="QWK").get(mode, "RMSE")
            k = int(grades[task]) if metric == "QWK" else None
            score = s.set_index("study_id").score
            if score.index.duplicated().any() or not t.study_id.isin(score.index).all():
                raise SystemExit(f"{name} {task}: scores must cover each TEST study once")
            p = score.loc[t.study_id].to_numpy(float)
            if not np.isfinite(p).all():
                raise SystemExit(f"{name} {task}: missing scores")
            y = t.label.to_numpy(float)
            bt = metric_draws(y, p, col[t.cluster_id].to_numpy(), draws, metric, k)
            boots[(name, task)] = bt
            lo, hi, valid = interval(bt)
            rows.append(dict(method=name, task=task, mode=mode, metric=metric, n_test=len(y),
                             n_positive=int(y.sum()) if metric == "AUROC" else None,
                             point=point(y, p, metric, k), lower=lo, upper=hi, valid_replicates=valid))
    per = pd.DataFrame(rows)

    paired, summary = [], []
    for pair in a.pair:
        x, y = pair.split(":")
        for t in sorted(set(per.task[per.method == x]) & set(per.task[per.method == y])):
            r = per[(per.task == t)].set_index("method")
            lo, hi, valid = interval(boots[(x, t)] - boots[(y, t)])
            paired.append(dict(a=x, b=y, task=t, mode=r.loc[x, "mode"], metric=r.loc[x, "metric"],
                               difference=r.loc[x, "point"] - r.loc[y, "point"], lower=lo, upper=hi, valid_replicates=valid))
    for mode, metric in (("classification", "AUROC"), ("ordinal", "QWK")):
        sub = per[per["mode"] == mode]
        keep = set(sub.task) if mode == "ordinal" or a.mean_tasks is None else set(sub.task) & set(a.mean_tasks)
        pt = lambda name, t: sub[(sub.method == name) & (sub.task == t)].point.iloc[0]
        for name in methods:
            tasks = sorted(set(sub.task[sub.method == name]) & keep)
            if not tasks:
                continue
            lo, hi, valid = interval(np.mean([boots[(name, t)] for t in tasks], 0))
            summary.append(dict(comparison=name, metric=metric, tasks=len(tasks),
                                point=float(np.mean([pt(name, t) for t in tasks])), lower=lo, upper=hi, valid_replicates=valid))
        for pair in a.pair:
            x, y = pair.split(":")
            tasks = sorted(set(sub.task[sub.method == x]) & set(sub.task[sub.method == y]) & keep)
            if not tasks:
                continue
            lo, hi, valid = interval(np.mean([boots[(x, t)] - boots[(y, t)] for t in tasks], 0))
            summary.append(dict(comparison=f"{x} minus {y}", metric=metric, tasks=len(tasks),
                                point=float(np.mean([pt(x, t) - pt(y, t) for t in tasks])), lower=lo, upper=hi,
                                valid_replicates=valid, tasks_a_higher=int(sum(pt(x, t) > pt(y, t) for t in tasks))))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_task.csv", index=False, float_format="%.17g")
    pd.DataFrame(paired).to_csv(out / "paired_per_task.csv", index=False, float_format="%.17g")
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False, float_format="%.17g")
    if a.save_draws:
        np.savez_compressed(out / "draws.npz", **{f"{m}|{t}": v for (m, t), v in boots.items()})
    print(pd.DataFrame(summary).to_string(), flush=True)


if __name__ == "__main__":
    main()
