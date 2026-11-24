#!/usr/bin/env python3
"""Write the manifest of every public dataset: data/manifests/<dataset>.csv.

    python scripts/make_manifests.py                      # every dataset with a folder set
    python scripts/make_manifests.py --datasets camus tmed2
"""
import argparse
import sys
import csv
import os
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import datasets_config  # noqa: E402
SPLITS = ROOT / "data/splits"
DATASETS = {
    "mimic_iv_echo": "mimic_iv_echo",
    "echonet_dynamic": "echonet_dynamic",
    "echonet_pediatric": "echonet_pediatric",
    "echonet_lvh": "echonet_lvh",
    "camus": "camus",
    "tmed2": "tmed2",
    "cardiacnet_asd": "cardiacnet",
    "cardiacnet_pah": "cardiacnet",
    "segrwma": "segrwma",
    "hmc_qu": "hmc_qu",
}
CARDIOBENCH = ["tmed2", "cardiacnet_asd", "cardiacnet_pah", "segrwma", "hmc_qu", "echonet_dynamic", "camus",
               "echonet_pediatric", "echonet_lvh"]
def mimic_manifest(root):
    records = next((p for p in (root / "echo-record-list.csv", root / "meta/echo-record-list.csv") if p.exists()), None)
    if records is None:
        raise SystemExit(f"mimic_iv_echo: no echo-record-list.csv in {root} or {root}/meta")
    studies = pd.read_csv(SPLITS / "mimic_iv_echo.csv", dtype=str).set_index("study_id")
    rows, seen = [], set()
    with open(records) as f:
        for r in csv.DictReader(f):
            p = r["dicom_filepath"]
            if p in seen or r["study_id"] not in studies.index:
                continue
            seen.add(p)
            path = root / p
            s = studies.loc[r["study_id"]]
            rows.append(dict(clip_id=Path(p).stem, study_id=r["study_id"], cluster_id=s.cluster_id,
                             split=s.split, path=str(path) if path.exists() else ""))
    d = pd.DataFrame(rows)
    n = d.groupby("study_id").size().reindex(studies.index, fill_value=0)
    diff = n[n != studies.n_clips.astype(int)]
    if len(diff):
        print(f"warning: mimic_iv_echo: {len(diff)} studies have a different number of files than in the paper, "
              f"for example study {diff.index[0]}: {diff.iloc[0]} on disk, {studies.n_clips[diff.index[0]]} in the paper")
    return d


def best_match(rel, candidates):
    def shared(p):
        n = 0
        for a, b in zip(reversed(rel.parts), reversed(p.parts)):
            if a != b:
                break
            n += 1
        return n
    if not candidates:
        return None
    scores = [shared(p) for p in candidates]
    best = max(scores)
    return candidates[scores.index(best)] if scores.count(best) == 1 else None


def split_manifest(name, root):
    d = pd.read_csv(SPLITS / f"{name}.csv", dtype=str)
    paths = [root / p for p in d.path]
    found = [p if p.exists() else None for p in paths]
    if any(p is None for p in found):
        index = {}
        for dirpath, _, files in os.walk(root):
            for f in files:
                index.setdefault(f, []).append(Path(dirpath) / f)
        found = [p or best_match(Path(rel), index.get(Path(rel).name, [])) for p, rel in zip(found, d.path)]
    d["path"] = [str(p) if p else "" for p in found]
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(datasets_config()), help="dataset folders (YAML)")
    ap.add_argument("--datasets", nargs="+", choices=list(DATASETS), help="datasets to write (default: every one with a folder)")
    ap.add_argument("--out", default=str(ROOT / "data/manifests"), help="output folder")
    ap.add_argument("--allow-missing", action="store_true", help="write the files that exist instead of stopping")
    a = ap.parse_args()

    roots = yaml.safe_load(open(a.config)) or {}
    names = a.datasets or [n for n, key in DATASETS.items() if roots.get(key)]
    if not names:
        raise SystemExit(f"no dataset folder set in {a.config}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    written = {}
    partial = set()
    for name in names:
        folder = roots.get(DATASETS[name])
        if not folder:
            raise SystemExit(f"{name}: set its folder ({DATASETS[name]}) in {a.config}")
        root = Path(folder).expanduser()
        d = mimic_manifest(root) if name == "mimic_iv_echo" else split_manifest(name, root)
        missing = d.path.eq("")
        if missing.any():
            msg = f"{name}: {missing.sum()} of {len(d)} files not found under {root}, for example {d.clip_id[missing].iloc[0]}"
            if not a.allow_missing:
                raise SystemExit(msg + " (use --allow-missing to write the rest)")
            print("warning:", msg)
            partial.add(name)
            d = d[~missing]
        d = d[["clip_id", "study_id", "cluster_id", "split", "path"]]
        d.to_csv(out / f"{name}.csv", index=False)
        written[name] = d
        counts = d.split.value_counts().reindex(["TRAIN", "VAL", "TEST"]).fillna(0).astype(int).to_dict()
        print(f"{name}: {len(d)} clips, {d.study_id.nunique()} studies, {counts} -> {out / f'{name}.csv'}")

    included = [n for n in CARDIOBENCH if n in written and n not in partial]
    if included:
        parts = []
        for n in included:
            p = written[n].copy()
            for c in ("clip_id", "study_id", "cluster_id"):
                p[c] = n + "/" + p[c]
            parts.append(p)
        d = pd.concat(parts, ignore_index=True)
        d.to_csv(out / "cardiobench.csv", index=False)
        print(f"cardiobench: {len(d)} clips from {', '.join(included)} -> {out / 'cardiobench.csv'}")
    if set(CARDIOBENCH) & partial:
        print(f"cardiobench.csv leaves out {', '.join(sorted(set(CARDIOBENCH) & partial))}: files missing")


if __name__ == "__main__":
    main()
