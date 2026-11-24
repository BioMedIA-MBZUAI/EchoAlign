#!/usr/bin/env python3
"""Linear probes on EchoPrime study features from EchoPrime's own section pooling.

    python scripts/probe_echoprime_sections.py --features embeddings/mimic_iv_echo/echoprime_sections.npz \
        --labels data/labels/mimic_iv_echo_probe.csv --cohort mimic_iv_echo \
        --out results/mimic_iv_echo/echoprime_sections
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis import probes  # noqa: E402
from eval.analysis.metrics import cluster_draws, rmse_ci, seed_from  # noqa: E402
from eval.analysis.ordinal import boundary_aurocs, ordinal_probe, qwk, qwk_ci  # noqa: E402
from eval.analysis.echoprime_sections import (check_section_map, load_section_map,  # noqa: E402
                                              section_map_table, task_vectors)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", required=True, help=".npz from scripts/echoprime_section_features.py")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--cohort", required=True, help="cohort in --section-map")
    ap.add_argument("--section-map", default=str(ROOT / "configs/echoprime_sections.yaml"))
    ap.add_argument("--name", default="echoprime_sections", help="output file prefix")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=None, help="BLAS threads")
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed-key", default=None, help="bootstrap seed key, as in scripts/probe.py")
    ap.add_argument("--missing", choices=["fail", "drop"], default="fail",
                    help="a study without features stops the run (fail) or is left out of the task (drop; counted in "
                         "the results file). Studies with only still images have no EchoPrime features")
    ap.add_argument("--probe-out", default=None,
                    help="also write the outputs in scripts/probe.py's file format, as model echoprime, to this folder")
    ap.add_argument("--probe-method", default="echoprime", help="model name of the --probe-out files")
    a = ap.parse_args()
    if a.threads:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=a.threads)
    name = a.name

    with np.load(a.features, allow_pickle=False) as f:
        study = f["study_id"].astype(str)
        vectors = f["sections"]
        sections = [str(s) for s in f["section_names"]]
    row = pd.Series(np.arange(len(study)), index=study)
    lab = pd.read_csv(a.labels, dtype={"study_id": str, "task": str, "split": str, "cluster_id": str})
    if "cluster_id" not in lab:
        lab["cluster_id"] = lab["study_id"]
    clusters = np.array(sorted(lab[lab.split == "TEST"].cluster_id.unique()))
    draws = cluster_draws(len(clusters), a.replicates, seed_from(a.seed_key or Path(a.labels).name))
    tasks = list(dict.fromkeys(lab.task))
    if a.tasks:
        tasks = [t for t in tasks if t in set(a.tasks)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    smap = load_section_map(a.section_map, a.cohort)
    check_section_map(smap, sections)
    missing = [t for t in tasks if t not in smap]
    if missing:
        raise SystemExit(f"tasks without an entry in {a.section_map}: {missing}")
    section_map_table({t: smap[t] for t in tasks}).to_csv(out / "task_section_map.csv", index=False)

    preds, results, selection, boundaries, regression, ordinal = [], [], {}, [], [], []
    for task in tasks:
        d = lab[lab.task == task]
        mode = d["mode"].fillna("classification").iloc[0] if "mode" in d else ("classification" if d.label.isin([0, 1]).all() else "regression")
        x, rep = task_vectors(smap[task], sections, vectors)
        has = d.study_id.isin(row.index)
        if not has.all() and a.missing == "fail":
            raise SystemExit(f"{task}: {int((~has).sum())} studies have no features (use --missing drop)")
        n_dropped = int((~has).sum())
        d = d[has]
        if any(d[d.split == s].empty for s in ("TRAIN", "VAL", "TEST")):
            print(f"{task}: skipped, no features for every study of a split ({n_dropped} studies dropped)", flush=True)
            continue
        expected, p_ge = None, {}
        part = {s: d[d.split == s] for s in ("TRAIN", "VAL", "TEST")}
        xs = {s: x[row[part[s].study_id].to_numpy()] for s in part}
        ys = {s: part[s].label.to_numpy() for s in part}
        ix = np.searchsorted(clusters, part["TEST"].cluster_id)
        if mode == "classification":
            fit = probes.classification_probe
            pred, rec = fit(xs["TRAIN"], ys["TRAIN"], xs["VAL"], ys["VAL"], xs["TEST"])
            point, metric, param = float(roc_auc_score(ys["TEST"].astype(int), pred)), "AUROC", rec["C"]
        elif mode == "ordinal":
            k = int(d.label.max()) + 1
            probs, pred, rec = ordinal_probe(xs["TRAIN"], ys["TRAIN"], xs["VAL"], ys["VAL"], xs["TEST"], k, task)
            expected = probs @ np.arange(k)
            p_ge = {f"p_ge_{j}": probs[:, j:].sum(1) for j in range(1, k)}
            ordinal.append(dict(method="echoprime", task=task, n=len(pred), grades=k,
                                **qwk_ci(ys["TEST"].astype(int), pred, k, ix, draws)))
            boundaries += [dict(method="echoprime", task=task, **b)
                           for b in boundary_aurocs(ys["TEST"].astype(int), probs, ix, draws)]
            point, metric, param = qwk(ys["TEST"].astype(int), pred, k), "QWK", rec["C"]
        else:
            pred, rec = probes.regression_probe(xs["TRAIN"], ys["TRAIN"], xs["VAL"], ys["VAL"], xs["TEST"])
            regression.append(dict(method="echoprime", task=task, n=len(pred), **rmse_ci(ys["TEST"], pred, ix, draws)))
            point, metric, param = float(mean_squared_error(ys["TEST"], pred) ** 0.5), "RMSE", rec["alpha"]
        selection[task] = dict(mode=mode, representation=rep, **rec)
        results.append(dict(task=task, mode=mode, representation=rep, metric=metric, point=point, parameter=param,
                            n_train=len(ys["TRAIN"]), n_val=len(ys["VAL"]), n_test=len(ys["TEST"]),
                            n_dropped_no_features=n_dropped))
        preds.append(pd.DataFrame(dict(study_id=part["TEST"].study_id.to_numpy(), task=task, mode=mode,
                                       y_true=ys["TEST"], y_pred=pred,
                                       expected_grade=np.nan if expected is None else expected, **p_ge)))
        print(f"{name} {task}: {rep}, {metric} {point:.4f}, parameter {param}", flush=True)

    pd.concat(preds).to_csv(out / f"{name}_test_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(results).to_csv(out / f"{name}_results.csv", index=False, float_format="%.17g")
    if regression:
        pd.DataFrame(regression).to_csv(out / f"{name}_regression.csv", index=False)
    if ordinal:
        pd.DataFrame(ordinal).to_csv(out / f"{name}_ordinal.csv", index=False)
    if boundaries:
        pd.DataFrame(boundaries).to_csv(out / f"{name}_ordinal_boundaries.csv", index=False)
    (out / f"{name}_selection.json").write_text(json.dumps(selection, indent=1, default=float) + "\n")
    print(f"{name}: {len(results)} tasks -> {out}")
    if a.probe_out:
        write_probe_files(out, name, Path(a.probe_out), a.probe_method)


def write_probe_files(src, name, out, method="echoprime"):
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(src / f"{name}_test_predictions.csv", dtype={"study_id": str}).assign(method=method)
    c = d[d["mode"] == "classification"].rename(columns={"y_pred": "score"})
    c[["study_id", "task", "method", "score"]].to_csv(out / f"{method}_test_scores.csv", index=False)
    r = d[d["mode"] == "regression"]
    r[["study_id", "task", "method", "y_true", "y_pred"]].to_csv(out / f"{method}_regression_predictions.csv",
                                                                 index=False)
    o = d[d["mode"] == "ordinal"].rename(columns={"y_true": "label", "y_pred": "prediction"})
    o["label"], o["prediction"] = o.label.astype(int), o.prediction.astype(int)
    o[["study_id", "task", "method", "label", "prediction", "expected_grade"]
      + [c for c in o.columns if c.startswith("p_ge_")]].to_csv(
        out / f"{method}_ordinal_predictions.csv", index=False)
    shutil.copyfile(src / f"{name}_selection.json", out / f"{method}_selection.json")
    for kind in ("regression", "ordinal", "ordinal_boundaries"):
        if (src / f"{name}_{kind}.csv").exists():
            shutil.copyfile(src / f"{name}_{kind}.csv", out / f"{method}_{kind}.csv")
    print(f"{method} probe files:", len(c), "classification,", len(r), "regression,", len(o), "ordinal rows")


if __name__ == "__main__":
    main()
