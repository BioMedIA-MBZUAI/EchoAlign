#!/usr/bin/env python3
"""Mean classification AUROC per label fraction for several models, with paired 95% intervals.

    python scripts/summarize_label_efficiency.py --labels results/mimic_iv_echo/compare_test_scores/paired_labels.csv \
        --universe results/mimic_iv_echo/compare_test_scores/universe.csv --seed-key mimic/patient_cluster/absolute \
        --method echoalign=results/mimic_iv_echo/label_efficiency/echoalign_label_efficiency_scores.csv \
        --method echoprime=... --out results/mimic_iv_echo/label_efficiency/summary
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.metrics import interval, metric_draws, point, cluster_draws, seed_from  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--seed-key", required=True)
    ap.add_argument("--method", action="append", required=True, help="NAME=CSV; the first is the reference")
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lab = pd.read_csv(a.labels, dtype={"study_id": str, "task": str, "cluster_id": str})
    lab = lab[lab.split.eq("TEST")]
    if "mode" in lab:
        lab = lab[lab["mode"].fillna("classification").eq("classification")]
    if "cluster_id" not in lab:
        lab["cluster_id"] = lab.study_id
    uni = pd.read_csv(a.universe, dtype=str)
    col = pd.Series(np.arange(len(uni)), index=uni["cluster_id" if "cluster_id" in uni else "study_id"].to_numpy())
    draws = cluster_draws(len(col), a.replicates, seed_from(a.seed_key))
    methods = dict(m.split("=", 1) for m in a.method)
    names = list(methods)
    ref = names[0]
    scores = {n: pd.read_csv(p, dtype={"study_id": str, "task": str}) for n, p in methods.items()}

    per, boots = [], {}
    for n, s in scores.items():
        for (fraction, seed, task), g in s.groupby(["fraction", "seed", "task"]):
            t = lab[lab.task == task]
            if t.empty:
                continue
            sc = g.set_index("study_id").score
            if not t.study_id.isin(sc.index).all():
                continue
            y, p = t.label.to_numpy(float), sc.loc[t.study_id].to_numpy(float)
            boots[(n, fraction, seed, task)] = metric_draws(y, p, col[t.cluster_id].to_numpy(), draws, "AUROC")
            per.append(dict(fraction=fraction, seed=seed, method=n, task=task, n_test=len(y), auroc=point(y, p, "AUROC")))
    per = pd.DataFrame(per)

    rows = []
    for fraction, f in per.groupby("fraction"):
        seeds = sorted(f.seed.unique())
        done = {t: set(zip(g.method, g.seed)) for t, g in f.groupby("task")}
        tasks = sorted(t for t, ms in done.items() if ms >= {(m, s) for m in names for s in seeds})
        if not tasks:
            continue
        task_mean = {n: {s: np.mean([boots[(n, fraction, s, t)] for t in tasks], 0) for s in seeds} for n in names}
        pt = {n: [f[(f.method == n) & (f.seed == s) & f.task.isin(tasks)].auroc.mean() for s in seeds] for n in names}
        for n in names:
            lo, hi, valid = interval(np.mean([task_mean[n][s] for s in seeds], 0))
            rows.append(dict(fraction=fraction, comparison=n, tasks=len(tasks), seeds=len(seeds), point=float(np.mean(pt[n])),
                             lower=lo, upper=hi, valid_replicates=valid, sd_across_seeds=float(np.std(pt[n], ddof=1))
                             if len(seeds) > 1 else 0.0))
        for n in names[1:]:
            diff = np.mean([task_mean[ref][s] - task_mean[n][s] for s in seeds], 0)
            lo, hi, valid = interval(diff)
            d = np.array(pt[ref]) - np.array(pt[n])
            rows.append(dict(fraction=fraction, comparison=f"{ref} minus {n}", tasks=len(tasks), seeds=len(seeds),
                             point=float(d.mean()), lower=lo, upper=hi, valid_replicates=valid,
                             sd_across_seeds=float(d.std(ddof=1)) if len(seeds) > 1 else 0.0))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per.to_csv(out / "per_task.csv", index=False, float_format="%.17g")
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False, float_format="%.17g")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
