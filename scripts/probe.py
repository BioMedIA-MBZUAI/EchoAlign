#!/usr/bin/env python3
"""Linear probes on frozen study features for one model and one cohort.

    python scripts/probe.py --method echoalign --embeddings embeddings/mimic_iv_echo/echoalign.npz \
        --labels data/labels/mimic_iv_echo_probe.csv --out results/mimic_iv_echo/probe

EchoPrime's paper probes use scripts/probe_echoprime_sections.py.
"""
import argparse
import functools
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import iter_npz, study_mean  # noqa: E402
from eval.embed import METHODS  # noqa: E402
from eval.analysis.metrics import cluster_draws, rmse_ci, seed_from  # noqa: E402
from eval.analysis.ordinal import boundary_aurocs, ordinal_probe, qwk_ci  # noqa: E402
from eval.analysis.probes import classification_probe, regression_probe  # noqa: E402


@functools.lru_cache(maxsize=None)
def pca_bases(weights, level):
    blob = torch.load(weights, map_location="cpu", weights_only=True)
    if f"pca_{level}" not in blob:
        raise SystemExit(f"{weights} has no pca_{level} bases")
    return {k: v.numpy().astype(np.float64) for k, v in blob[f"pca_{level}"].items()}


def pca832(vectors, weights, level):
    p = pca_bases(weights, level)
    x = np.asarray(vectors, np.float64)
    return np.einsum("nsd,skd->nsk", x - p["mean"][None], p["components"]).reshape(len(x), -1)


def study_features(method, emb, weights, level):
    if method == "echoalign":
        if level == "study":
            return pd.DataFrame(pca832(emb["vectors"], weights, "study"), index=emb["study_id"])
        clips = np.concatenate([emb["global_vector"][:, None], emb["structures"]], axis=1)
        return study_mean(pca832(clips, weights, "clip"), emb["study_id"])
    if method == "echoclip":
        clip = emb["frame_features"].sum(1)                     # padded frames are zero
        clip = clip / np.maximum(np.linalg.norm(clip, axis=1, keepdims=True), 1e-12)
        return study_mean(clip, emb["study_id"])
    return study_mean(emb["features"], emb["study_id"])


def train_subset(train, task, fraction, seed):
    """Stratified TRAIN subset: within each label value, the first round(fraction * n) studies ranked by
    SHA-256 of "<task>|<seed>|<study_id>". The ranking ignores the fraction and the model, so subsets of one seed are
    nested (1% within 5% within ...) and every model gets the same studies."""
    keep = []
    for _, g in train.groupby("label", sort=True):
        rank = g.study_id.map(lambda s: hashlib.sha256(f"{task}|{seed}|{s}".encode()).hexdigest())
        keep.append(g.loc[rank.sort_values(kind="stable").index[:int(np.floor(fraction * len(g) + 0.5))]])
    return train[train.index.isin(pd.concat(keep).index)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--embeddings", required=True, help=".npz from scripts/embed.py (EchoAlign: the study file, "
                                                        "or the _clips file with --level clip)")
    ap.add_argument("--labels", required=True, help="label file")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--weights", default=str(ROOT / "weights/EchoAlign.pt"))
    ap.add_argument("--level", choices=["study", "clip"], default="study", help="EchoAlign feature level")
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed-key", default=None)
    ap.add_argument("--threads", type=int, default=None,
                    help="BLAS threads; logistic fits that stop at tolerance depend slightly on this (paper: 4 for "
                         "every model, with OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS set to 4)")
    ap.add_argument("--tasks", nargs="+", default=None, help="tasks to run (default: all). Bootstrap draws still use every study of the label file")
    ap.add_argument("--missing", choices=["fail", "drop"], default="fail",
                    help="a study without an embedding stops the run (fail) or is left out of that task's TRAIN, VAL "
                         "and TEST rows (drop; counted in the selection file). EchoPrime embeds videos only, so a "
                         "study with only still images has no EchoPrime embedding")
    a = ap.parse_args()
    return ap, a


def probe_one(ap, a):
    tag = a.method
    if a.threads:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=a.threads)

    feats = pd.concat([study_features(a.method, e, a.weights, a.level) for e in iter_npz(a.embeddings)])
    assert feats.index.is_unique, "a study appears in more than one shard"
    lab = pd.read_csv(a.labels, dtype={"study_id": str, "task": str, "split": str, "cluster_id": str})
    if "cluster_id" not in lab:
        lab["cluster_id"] = lab["study_id"]
    test = lab[lab.split == "TEST"]
    clusters = np.array(sorted(test.cluster_id.unique()))
    draws = cluster_draws(len(clusters), a.replicates, seed_from(a.seed_key or Path(a.labels).name))

    scores, regression, reg_preds, selection = [], [], [], {}
    ordinal, boundaries, grades = [], [], []
    if a.tasks and set(a.tasks) - set(lab.task):
        ap.error(f"tasks not in {a.labels}: {sorted(set(a.tasks) - set(lab.task))}")
    for task, d in lab.groupby("task", sort=False):
        if a.tasks and task not in a.tasks:
            continue
        has = d.study_id.isin(feats.index)
        if not has.all() and a.missing == "fail":
            raise SystemExit(f"{task}: {int((~has).sum())} studies have no embedding (use --missing drop)")
        dropped = {s: int((~has & (d.split == s)).sum()) for s in ("TRAIN", "VAL", "TEST")}
        d = d[has]
        if any(d[d.split == s].empty for s in ("TRAIN", "VAL", "TEST")):
            print(f"{tag} {task}: skipped, no embedding for every study of a split (dropped {dropped})", flush=True)
            continue
        mode = d["mode"].fillna("classification").iloc[0] if "mode" in d else ("classification" if d.label.isin([0, 1]).all() else "regression")
        part = {s: d[d.split == s] for s in ("TRAIN", "VAL", "TEST")}
        if mode == "classification" and any(part[s].label.nunique() < 2 for s in ("TRAIN", "TEST")):
            print(f"{tag} {task}: skipped, TRAIN or TEST has one class", flush=True)
            continue
        x = {s: feats.loc[part[s].study_id].to_numpy() for s in part}
        y = {s: part[s].label.to_numpy() for s in part}
        ix = np.searchsorted(clusters, part["TEST"].cluster_id)
        if mode == "ordinal":
            k = int(d.label.max()) + 1
            probs, pred, record = ordinal_probe(x["TRAIN"], y["TRAIN"], x["VAL"], y["VAL"], x["TEST"], k, task)
            yt = y["TEST"].astype(int)
            ordinal.append(dict(method=a.method, task=task, n=len(yt), grades=k, **qwk_ci(yt, pred, k, ix, draws)))
            boundaries += [dict(method=a.method, task=task, **b) for b in boundary_aurocs(yt, probs, ix, draws)]
            grades.append(pd.DataFrame(dict(study_id=part["TEST"].study_id, task=task, method=a.method, label=yt,
                                            prediction=pred, expected_grade=probs @ np.arange(k),
                                            **{f"p_ge_{j}": probs[:, j:].sum(1) for j in range(1, k)})))
        elif mode == "classification":
            pred, record = classification_probe(x["TRAIN"], y["TRAIN"], x["VAL"], y["VAL"], x["TEST"])
            scores.append(pd.DataFrame(dict(study_id=part["TEST"].study_id, task=task, method=a.method, score=pred)))
        else:
            pred, record = regression_probe(x["TRAIN"], y["TRAIN"], x["VAL"], y["VAL"], x["TEST"])
            regression.append(dict(method=a.method, task=task, n=len(pred), **rmse_ci(y["TEST"], pred, ix, draws)))
            reg_preds.append(pd.DataFrame(dict(study_id=part["TEST"].study_id, task=task, method=a.method,
                                               y_true=y["TEST"], y_pred=pred)))
        selection[task] = dict(mode=mode, n_train=len(y["TRAIN"]), n_val=len(y["VAL"]), n_test=len(y["TEST"]),
                               dropped_no_embedding=dropped, **record)
        print(f"{a.method} {task}: {mode}, " + (f"alpha={record['alpha']}" if mode == "regression" else f"C={record['C']}"),
              flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if scores:
        pd.concat(scores).to_csv(out / f"{tag}_test_scores.csv", index=False)
    if regression:
        pd.DataFrame(regression).to_csv(out / f"{tag}_regression.csv", index=False)
        pd.concat(reg_preds).to_csv(out / f"{tag}_regression_predictions.csv", index=False, float_format="%.17g")
    if ordinal:
        pd.DataFrame(ordinal).to_csv(out / f"{tag}_ordinal.csv", index=False)
        pd.DataFrame(boundaries).to_csv(out / f"{tag}_ordinal_boundaries.csv", index=False)
        pd.concat(grades).to_csv(out / f"{tag}_ordinal_predictions.csv", index=False)
    (out / f"{tag}_selection.json").write_text(json.dumps(selection, indent=1) + "\n")
    print(f"{tag}: {len(selection)} tasks -> {out}")


def main():
    ap, a = parse_args()
    probe_one(ap, a)


if __name__ == "__main__":
    main()
