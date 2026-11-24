#!/usr/bin/env python3
"""Classification AUROC of each model within patient subgroups of one cohort, with paired intervals.

    python scripts/subgroups.py --cohort mimic_iv_echo --labels results/mimic_iv_echo/compare_test_scores/paired_labels.csv \
        --universe results/mimic_iv_echo/compare_test_scores/universe.csv --seed-key mimic/patient_cluster/absolute \
        --covariates <covariate table> --method echoalign=... --method echoprime=... --out results/mimic_iv_echo/subgroups
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.metrics import interval, metric_draws, point, cluster_draws, seed_from  # noqa: E402


def levels(cov, spec):
    if "threshold" in spec:
        cols = list(spec["threshold"])
        above = np.zeros(len(cov), bool)
        for c, limit in spec["threshold"].items():
            above |= cov[c].to_numpy(float) >= limit
        lev = pd.Series(np.where(above, spec["above"], spec["below"]), index=cov.index, dtype=object)
        return lev[cov[cols[0]].notna()]
    v = cov[spec["column"]]
    if "bins" in spec:
        lev = pd.cut(v.astype(float), [-np.inf, *spec["bins"], np.inf], right=False, labels=spec["labels"]).astype(object)
    elif "map" in spec:
        lev = v.map(spec["map"])
    else:
        lev = v.astype(object)
    lev = lev.dropna()
    if "rank" in spec:
        order = lev.value_counts().index
        lev = lev.map({s: f"{spec['rank']} {i + 1:02d}" for i, s in enumerate(order)})
    return lev


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True, help="cohort in --config")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--seed-key", required=True)
    ap.add_argument("--covariates", required=True)
    ap.add_argument("--method", action="append", required=True, help="NAME=CSV; the first is the reference")
    ap.add_argument("--config", default=str(ROOT / "configs/subgroups.yaml"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--min-class", type=int, default=10)
    ap.add_argument("--small", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lab = pd.read_csv(a.labels, dtype={"study_id": str, "task": str, "cluster_id": str})
    lab = lab[lab.split.eq("TEST") & lab.get("mode", pd.Series("classification", index=lab.index)).eq("classification")]
    if "cluster_id" not in lab:
        lab["cluster_id"] = lab.study_id
    uni = pd.read_csv(a.universe, dtype=str)
    col = pd.Series(np.arange(len(uni)), index=uni["cluster_id" if "cluster_id" in uni else "study_id"].to_numpy())
    draws = cluster_draws(len(col), a.replicates, seed_from(a.seed_key))
    methods = dict(m.split("=", 1) for m in a.method)
    names = list(methods)
    ref = names[0]
    scores = {n: pd.read_csv(p, dtype={"study_id": str}).set_index(["study_id", "task"])["score"] for n, p in methods.items()}

    data = {}
    for task, t in lab.groupby("task"):
        keys = pd.MultiIndex.from_arrays([t.study_id, t.task])
        if not all(keys.isin(s.index).all() for s in scores.values()):
            continue
        data[task] = dict(sid=t.study_id.to_numpy(), y=t.label.to_numpy(float), ix=col[t.cluster_id].to_numpy(),
                          p={n: s.loc[keys].to_numpy(float) for n, s in scores.items()})

    cov = pd.read_csv(a.covariates, dtype={"study_id": str}).set_index("study_id")
    config = yaml.safe_load(open(a.config))[a.cohort]
    test = set(lab.study_id)

    def evaluate(task, mask):
        d = data[task]
        y = d["y"][mask]
        if (y == 1).sum() < a.min_class or (y == 0).sum() < a.min_class:
            return None
        return {n: (point(y, d["p"][n][mask], "AUROC"), metric_draws(y, d["p"][n][mask], d["ix"][mask], draws, "AUROC"))
                for n in names}

    counts, task_rows, res, pair, inter = [], [], [], [], []
    for covariate, spec in config.items():
        lev = levels(cov, spec)
        lev = lev[lev.index.isin(test)]
        per_level = {}
        for level in sorted(lev.unique(), key=str):
            studies = set(lev.index[lev.eq(level)])
            small = len(studies) < a.small
            counts.append(dict(covariate=covariate, level=level, n_studies=len(studies), small_level=small))
            ev = {}
            for task, d in data.items():
                mask = np.isin(d["sid"], list(studies))
                r = evaluate(task, mask)
                y = d["y"][mask]
                for n in names:
                    pt, bt = r[n] if r else (np.nan, None)
                    lo, hi, inv = interval(bt) if r else (np.nan, np.nan, np.nan)
                    task_rows.append(dict(covariate=covariate, level=level, task=task, model=n, n_studies=int(mask.sum()),
                                          n_pos=int(y.sum()), n_neg=int((1 - y).sum()), evaluable=r is not None,
                                          auroc=pt, lower=lo, upper=hi, valid_replicates=inv))
                if r:
                    ev[task] = r
            per_level[level] = (ev, small, len(studies))
        big = [L for L, (_, small, _) in per_level.items() if not small]
        fixed = sorted(set.intersection(*[set(per_level[L][0]) for L in big])) if big else []
        everyone = {}
        if fixed:
            for task in fixed:
                everyone[task] = evaluate(task, np.ones(len(data[task]["y"]), bool))
            per_level["All"] = ({t: r for t, r in everyone.items() if r}, False, len(test))
        stored = {}
        for level, (ev, small, n_st) in per_level.items():
            sets = [("fixed", fixed)] if level == "All" else [("variable", sorted(ev))] + ([] if small else [("fixed", fixed)])
            for ts_name, ts in sets:
                ts = [t for t in ts if t in ev]
                base = dict(covariate=covariate, level=level, n_studies=n_st, task_set=ts_name, n_tasks=len(ts))
                if not ts:
                    continue
                mp = {n: float(np.mean([ev[t][n][0] for t in ts])) for n in names}
                mb = {n: np.mean([ev[t][n][1] for t in ts], 0) for n in names}
                for n in names:
                    lo, hi, inv = interval(mb[n])
                    res.append(dict(base, model=n, mean_auroc=mp[n], lower=lo, upper=hi, valid_replicates=inv))
                for n in names[1:]:
                    lo, hi, inv = interval(mb[ref] - mb[n])
                    pair.append(dict(base, model=ref, comparator=n, difference=mp[ref] - mp[n], lower=lo, upper=hi,
                                     valid_replicates=inv, model_mean=mp[ref], comparator_mean=mp[n]))
                    if ts_name == "fixed" and level != "All":
                        stored[(level, n)] = (mp[ref] - mp[n], mb[ref] - mb[n])
        for n in names[1:]:
            ks = [L for L in per_level if (L, n) in stored]
            if len(ks) < 2:
                continue
            pts = np.array([stored[(L, n)][0] for L in ks])
            bs = np.stack([stored[(L, n)][1] for L in ks])
            bs = bs[:, np.isfinite(bs).all(0)]
            w = 1 / bs.var(1, ddof=1)
            q = float(np.sum(w * (pts - np.sum(w * pts) / w.sum()) ** 2))
            cen = bs - bs.mean(1, keepdims=True)
            qb = (w[:, None] * (cen - (w[:, None] * cen).sum(0) / w.sum()) ** 2).sum(0)
            hi_i, lo_i = int(np.argmax(pts)), int(np.argmin(pts))
            rng = bs[hi_i] - bs[lo_i]
            inter.append(dict(covariate=covariate, comparator=n, n_levels=len(ks), n_tasks_fixed=len(fixed),
                              level_max=ks[hi_i], diff_max=pts[hi_i], level_min=ks[lo_i], diff_min=pts[lo_i],
                              range_diff=pts[hi_i] - pts[lo_i], range_lower=float(np.quantile(rng, 0.025)),
                              range_upper=float(np.quantile(rng, 0.975)), Q=q, df=len(ks) - 1,
                              p_chi2=float(stats.chi2.sf(q, len(ks) - 1)), p_boot=float((qb >= q).mean()),
                              valid_replicates=bs.shape[1]))
        print(f"{covariate}: {len(per_level) - bool(fixed)} levels, fixed task set {len(fixed)}", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("level_counts", counts), ("task_auroc", task_rows), ("results", res), ("paired", pair),
                       ("interaction", inter)):
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False, float_format="%.17g")
    print(f"{len(data)} tasks -> {out}")


if __name__ == "__main__":
    main()
