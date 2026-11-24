#!/usr/bin/env python3
"""Label efficiency: classification AUROC when the linear probes see only part of the TRAIN labels.

    python scripts/label_efficiency.py --method echoalign --embeddings embeddings/mimic_iv_echo/echoalign.npz \
        --labels data/labels/mimic_iv_echo_probe.csv --out results/mimic_iv_echo/label_efficiency

    python scripts/label_efficiency.py --method echoprime --echoprime-sections embeddings/mimic_iv_echo/echoprime_sections.npz \
        --cohort mimic_iv_echo --labels data/labels/mimic_iv_echo_probe.csv \
        --missing drop --out results/mimic_iv_echo/label_efficiency
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval.analysis.data import iter_npz  # noqa: E402
from eval.analysis.echoprime_sections import check_section_map, load_section_map, task_vectors  # noqa: E402
from eval.analysis.probes import classification_probe  # noqa: E402
from eval.embed import METHODS  # noqa: E402
from probe import study_features, train_subset  # noqa: E402

FRACTIONS = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--embeddings", help=".npz from scripts/embed.py")
    ap.add_argument("--echoprime-sections", help=".npz from scripts/echoprime_section_features.py (EchoPrime only)")
    ap.add_argument("--cohort", help="cohort in --section-map (with --echoprime-sections)")
    ap.add_argument("--section-map", default=str(ROOT / "configs/echoprime_sections.yaml"))
    ap.add_argument("--labels", required=True)
    ap.add_argument("--weights", default=str(ROOT / "weights/EchoAlign.pt"))
    ap.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--missing", choices=["fail", "drop"], default="fail")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--name", default=None, help="output prefix (default: the method, or echoprime_sections_all_sections)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=a.threads)
    if (a.embeddings is None) == (a.echoprime_sections is None):
        ap.error("give exactly one of --embeddings and --echoprime-sections")
    name = a.name or (a.method if a.embeddings else "echoprime_sections_all_sections")

    lab = pd.read_csv(a.labels, dtype={"study_id": str, "task": str, "split": str})
    if "mode" in lab:
        lab = lab[lab["mode"].fillna("classification").eq("classification")]
    lab = lab[lab.groupby("task").label.transform(lambda v: v.isin([0, 1]).all())]
    tasks = [t for t in dict.fromkeys(lab.task) if not a.tasks or t in a.tasks]

    if a.embeddings:
        feats = pd.concat([study_features(a.method, e, a.weights, "study") for e in iter_npz(a.embeddings)])
        features = lambda task: feats
    else:
        with np.load(a.echoprime_sections, allow_pickle=False) as f:
            study, vectors = f["study_id"].astype(str), f["sections"]
            sections = [str(s) for s in f["section_names"]]
        smap = load_section_map(a.section_map, a.cohort)
        check_section_map(smap, sections)
        features = lambda task: pd.DataFrame(task_vectors(smap[task], sections, vectors)[0], index=study)

    scores, fits = [], []
    for task in tasks:
        d = lab[lab.task == task]
        x_all = features(task)
        has = d.study_id.isin(x_all.index)
        if not has.all() and a.missing == "fail":
            raise SystemExit(f"{task}: {int((~has).sum())} studies have no features (use --missing drop)")
        d = d[has]
        part = {s: d[d.split == s] for s in ("TRAIN", "VAL", "TEST")}
        if any(part[s].label.nunique() < 2 for s in ("TRAIN", "TEST")):
            print(f"{name} {task}: skipped, TRAIN or TEST has one class", flush=True)
            continue
        x = {s: x_all.loc[part[s].study_id].to_numpy() for s in ("VAL", "TEST")}
        for fraction in sorted(a.fractions):
            for seed in (a.seeds if fraction < 1 else a.seeds[:1]):
                train = part["TRAIN"] if fraction == 1 else train_subset(part["TRAIN"], task, fraction, seed)
                row = dict(task=task, fraction=round(fraction, 6), seed=seed, n_train=len(train), n_pos=int(train.label.sum()))
                if train.label.nunique() < 2 or train.label.value_counts().min() < 2:
                    fits.append(dict(row, status="skipped: fewer than 2 studies of a class"))
                    continue
                pred, record = classification_probe(x_all.loc[train.study_id].to_numpy(), train.label.to_numpy(),
                                                    x["VAL"], part["VAL"].label.to_numpy(), x["TEST"])
                auroc = float(roc_auc_score(part["TEST"].label.astype(int), pred))
                fits.append(dict(row, status="ok", C=record["C"], test_auroc=auroc))
                scores.append(pd.DataFrame(dict(study_id=part["TEST"].study_id.to_numpy(), task=task, fraction=round(fraction, 6),
                                                seed=seed, score=pred)))
        print(f"{name} {task}: {sum(f['task'] == task and f['status'] == 'ok' for f in fits)} fits", flush=True)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(scores).to_csv(out / f"{name}_label_efficiency_scores.csv", index=False, float_format="%.17g")
    pd.DataFrame(fits).to_csv(out / f"{name}_label_efficiency_fits.csv", index=False, float_format="%.17g")
    print(f"{name}: {len(fits)} fits -> {out}")


if __name__ == "__main__":
    main()
