#!/usr/bin/env python3
"""Write the split files of the nine CardioBench datasets, data/splits/<dataset>.csv.

CAMUS, EchoNet-Dynamic, EchoNet-Pediatric and EchoNet-LVH come from each dataset's official split files, in the
dataset folders set in configs/datasets.yaml. TMED-2, CardiacNet, SegRWMA and HMC-QU come from the CardioBench
submodule.

    python scripts/make_cardiobench_splits.py                         # writes data/splits/
    python scripts/make_cardiobench_splits.py --out <scratch folder>  # to compare with data/splits/ first
    python scripts/make_cardiobench_splits.py --datasets segrwma --segrwma all
"""
import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import datasets_config  # noqa: E402

CB = ROOT / "benchmarks/CardioBench/data/splits"
SPLIT = {"train": "TRAIN", "val": "VAL", "test": "TEST"}
COLUMNS = ["clip_id", "study_id", "cluster_id", "split", "path"]
PEDIATRIC_FOLDS = {"0": "test", "1": "val"}
LVH = ["IVSd", "LVIDd", "LVPWd"]
CAMUS = {"training": "train", "validation": "val", "testing": "test"}
SOURCES = {
    "hmc_qu": ("hmcqu_split.csv", "data/raw/HMC_QU/"),
    "segrwma": ("segrwma_split.csv", "data/raw/Regional_wall_motion_abnormality/"),
    "cardiacnet_asd": ("cardiacnet/cardiacnet_asd_split.csv", "data/raw/CardiacNet/"),
    "cardiacnet_pah": ("cardiacnet/cardiacnet_pah_split.csv", "data/raw/CardiacNet/"),
    "tmed2": ("tmed2/per_image.csv", "data/raw/TMED_2/"),
}
OFFICIAL = ["camus", "echonet_dynamic", "echonet_pediatric", "echonet_lvh"]


def relative(paths, prefix, name):
    bad = ~paths.str.startswith(prefix)
    if bad.any():
        raise SystemExit(f"{name}: {bad.sum()} paths do not start with {prefix}, for example {paths[bad].iloc[0]}")
    return paths.str[len(prefix):]


def official(name, root):
    if name == "camus":
        d = pd.concat([pd.read_csv(root / f"database_split/subgroup_{f}.txt", header=None, names=["study_id"],
                                   dtype=str).assign(split=s) for f, s in CAMUS.items()])
        d = d.merge(pd.DataFrame({"view": ["2CH", "4CH"]}), how="cross")
        clip = d.study_id + "_" + d.view
        return d.assign(clip_id=clip, path="database_nifti/" + d.study_id + "/" + clip + "_half_sequence.nii.gz")
    if name == "echonet_dynamic":
        d = pd.read_csv(root / "FileList.csv", dtype=str)
        return d.assign(clip_id=d.FileName, study_id=d.FileName, split=d.Split.str.lower(),
                        path="Videos/" + d.FileName + ".avi")
    if name == "echonet_pediatric":
        d = pd.concat([pd.read_csv(root / v / "FileList.csv", dtype=str).assign(path=v + "/Videos/")
                       for v in ("A4C", "PSAX")])
        return d.assign(clip_id=d.FileName, study_id=d.FileName.str.rsplit("-", n=1).str[0],
                        split=d.Split.map(PEDIATRIC_FOLDS).fillna("train"), path=d.path + d.FileName)
    d = pd.read_csv(root / "MeasurementsList.csv", dtype=str)
    d = d[d.Calc.isin(LVH)].drop_duplicates("HashedFileName")
    return d.assign(clip_id=d.HashedFileName, study_id=d.HashedFileName, split=d.split.str.lower(),
                    path="videos/" + d.HashedFileName + ".avi")


def build(name, cb, folder, segrwma):
    if name in OFFICIAL:
        d = official(name, folder(name))
        d = d.assign(cluster_id=d.study_id)
    else:
        file, prefix = SOURCES[name]
        d = pd.read_csv(cb / file, dtype=str, keep_default_na=False)
    if name == "hmc_qu":
        d = d.assign(clip_id=d.unique_id, study_id=d.unique_id, cluster_id=d.patient_id)
    elif name == "segrwma":
        d = d[d.modality == "2D"]
        d = d.assign(clip_id=d.patient_view_id, study_id=d.patient_id, cluster_id=d.patient_id)
        if segrwma == "video":
            root = folder(name)
            d = d[[not np.isin(np.asanyarray(nib.load(root / p).dataobj), (0, 1)).all()
                   for p in relative(d.path, prefix, name)]]
    elif name.startswith("cardiacnet"):
        d = d.assign(clip_id=d.unique_id, study_id=d.unique_id, cluster_id=d.unique_id)
    elif name == "tmed2":
        d = d.assign(clip_id=d.query_key, study_id=d.query_key.str.rsplit("_", n=1).str[0], path=d.file_path)
        d = d.assign(cluster_id=d.study_id)
        study = pd.read_csv(cb / "tmed2/per_study.csv", dtype=str).set_index("study_id").split
        moved = d.split.values != study.reindex(d.study_id).values
        if moved.any():
            raise SystemExit(f"tmed2: {moved.sum()} images have another split than tmed2/per_study.csv "
                             f"(or a study not in it), for example {d.clip_id[moved].iloc[0]}")
    unknown = ~d.split.isin(list(SPLIT))
    if unknown.any():
        raise SystemExit(f"{name}: unknown split values {sorted(set(d.split[unknown]))}")
    if not d.clip_id.is_unique:
        raise SystemExit(f"{name}: clip_id is not unique, for example {d.clip_id[d.clip_id.duplicated()].iloc[0]}")
    d = d.assign(split=d.split.map(SPLIT))
    if name not in OFFICIAL:
        d = d.assign(path=relative(d.path, prefix, name))
    return d[COLUMNS].sort_values(["split", "study_id", "clip_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(datasets_config()), help="dataset folders (YAML)")
    ap.add_argument("--cardiobench", default=str(CB), help="CardioBench's data/splits folder")
    ap.add_argument("--datasets", nargs="+", choices=OFFICIAL + list(SOURCES), default=OFFICIAL + list(SOURCES))
    ap.add_argument("--segrwma", choices=["video", "all"], default="video",
                    help="video: drop the 2D _image files that hold only 0/1 mask values (515 clips); "
                         "all: keep every 2D clip of CardioBench (529 clips)")
    ap.add_argument("--out", default=str(ROOT / "data/splits"), help="output folder")
    a = ap.parse_args()

    roots = yaml.safe_load(open(a.config)) or {}

    def folder(name):
        if not roots.get(name):
            raise SystemExit(f"{name}: set its folder in {a.config}")
        return Path(roots[name]).expanduser()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in a.datasets:
        d = build(name, Path(a.cardiobench), folder, a.segrwma)
        d.to_csv(out / f"{name}.csv", index=False)
        n = d.split.value_counts().reindex(["TRAIN", "VAL", "TEST"]).fillna(0).astype(int).to_dict()
        print(f"{name}: {len(d)} clips, {d.study_id.nunique()} studies, {d.cluster_id.nunique()} clusters {n} "
              f"-> {out / (name + '.csv')}")


if __name__ == "__main__":
    main()
