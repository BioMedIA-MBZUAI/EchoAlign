#!/usr/bin/env python3
"""Write the CardioBench label files from each dataset's official label files and the CardioBench submodule.

Dataset folders come from configs/datasets.yaml; splits and clusters come from data/splits/<dataset>.csv
(scripts/make_cardiobench_splits.py).

    python scripts/make_cardiobench_labels.py                      # every dataset with a folder set
    python scripts/make_cardiobench_labels.py --datasets cardiacnet_asd cardiacnet_pah
    python scripts/make_cardiobench_labels.py --out <scratch folder>  # to compare with data/labels/ first
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import datasets_config  # noqa: E402

CB = ROOT / "benchmarks/CardioBench/data/splits"
TASKS = {
    "aortic_stenosis_tmed2_public": ("tmed2", "classification"),
    "asd_cardiacnet_public": ("cardiacnet_asd", "classification"),
    "pah_cardiacnet_public": ("cardiacnet_pah", "classification"),
    "rwma_public": ("segrwma", "classification"),
    "stemi_hmc_qu_public": ("hmc_qu", "classification"),
    "ejection_fraction_public": ("echonet_dynamic", "regression"),
    "ejection_fraction_camus_public": ("camus", "regression"),
    "ejection_fraction_pediatric_public": ("echonet_pediatric", "regression"),
    "ivsd_lvh_public": ("echonet_lvh", "regression"),
    "lvidd_lvh_public": ("echonet_lvh", "regression"),
    "lvpwd_lvh_public": ("echonet_lvh", "regression"),
}
AS = {"no_AS": 0, "mild_AS": 0, "mildtomod_AS": 0, "moderate_AS": 1, "severe_AS": 1}
LVH = {"ivsd_lvh_public": "IVSd", "lvidd_lvh_public": "LVIDd", "lvpwd_lvh_public": "LVPWd"}
ORDER = {"TRAIN": 0, "VAL": 1, "TEST": 2}
FOLDER = {"cardiacnet_asd": "cardiacnet", "cardiacnet_pah": "cardiacnet"}


def labels(task, folder, cb, studies):
    if task == "aortic_stenosis_tmed2_public":
        g = pd.read_csv(folder("tmed2") / "labels_per_image.csv", dtype=str)
        g = g.assign(s=g.query_key.str.rsplit("_", n=1).str[0]).drop_duplicates("s").set_index("s").diagnosis_label
        s = pd.read_csv(cb / "tmed2/per_study.csv", dtype=str).study_id
        return s, g.reindex(s).map(AS)
    if task in ("asd_cardiacnet_public", "pah_cardiacnet_public"):
        col = task[:3].upper()
        d = pd.read_csv(cb / f"cardiacnet/cardiacnet_{col.lower()}_split.csv", dtype=str)
        return d.unique_id, d[col]
    if task == "rwma_public":
        d = pd.read_csv(cb / "segrwma_split.csv", dtype=str)
        d = d[(d.modality == "2D") & d.patient_id.isin(studies.index)].drop_duplicates("patient_id")
        return d.patient_id, d.abnormal
    if task == "stemi_hmc_qu_public":
        d = pd.read_csv(cb / "hmcqu_split.csv", dtype=str)
        return d.unique_id, d.STEMI
    if task == "ejection_fraction_public":
        d = pd.read_csv(folder("echonet_dynamic") / "FileList.csv", dtype=str)
        return d.FileName, d.EF
    if task == "ejection_fraction_camus_public":
        s = studies.index.to_series()
        ef = [next(line.split(":")[1].strip() for line in open(folder("camus") / f"database_nifti/{p}/Info_2CH.cfg")
                   if line.startswith("EF:")) for p in s]
        return s, ef
    if task == "ejection_fraction_pediatric_public":
        root = folder("echonet_pediatric")
        d = pd.concat([pd.read_csv(root / v / "FileList.csv", dtype=str) for v in ("A4C", "PSAX")])
        d = d.assign(s=d.FileName.str.rsplit("-", n=1).str[0]).drop_duplicates("s")
        return d.s, d.EF
    d = pd.read_csv(folder("echonet_lvh") / "MeasurementsList.csv", dtype=str)
    d = d[d.Calc == LVH[task]].drop_duplicates("HashedFileName")
    return d.HashedFileName, d.CalcValue


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(datasets_config()), help="dataset folders (YAML)")
    ap.add_argument("--cardiobench", default=str(CB), help="CardioBench's data/splits folder")
    ap.add_argument("--splits", default=str(ROOT / "data/splits"), help="folder with <dataset>.csv split files")
    ap.add_argument("--out", default=str(ROOT / "data/labels"))
    datasets = sorted({d for d, _ in TASKS.values()})
    ap.add_argument("--datasets", nargs="+", choices=datasets,
                    help="datasets to label (default: every one with a folder set)")
    a = ap.parse_args()

    roots = yaml.safe_load(open(a.config)) or {}
    names = a.datasets or [d for d in datasets if roots.get(FOLDER.get(d, d))]
    if not names:
        raise SystemExit(f"no CardioBench dataset folder set in {a.config}")

    def folder(name):
        if not roots.get(name):
            raise SystemExit(f"{name}: set its folder in {a.config}")
        return Path(roots[name]).expanduser()

    rows = []
    for task, (dataset, mode) in TASKS.items():
        if dataset not in names:
            print(f"{task} ({dataset}): skipped, dataset not selected")
            continue
        splits = pd.read_csv(Path(a.splits) / f"{dataset}.csv", dtype=str)
        studies = splits.drop_duplicates("study_id").set_index("study_id")
        uuid, y = labels(task, folder, Path(a.cardiobench), studies)
        lab = pd.DataFrame({"study_uuid": list(uuid), "y": pd.to_numeric(pd.Series(list(y))).astype(float)})
        if lab.y.isna().any():
            raise SystemExit(f"{task}: {lab.y.isna().sum()} studies without a label, "
                             f"for example {lab.study_uuid[lab.y.isna()].iloc[0]}")
        unknown = ~lab.study_uuid.isin(studies.index)
        if unknown.any():
            raise SystemExit(f"{task}: {unknown.sum()} labeled studies not in {a.splits}/{dataset}.csv, "
                             f"for example {lab.study_uuid[unknown].iloc[0]}")
        lab = lab.assign(split=studies.split.reindex(lab.study_uuid).values)
        lab = lab.assign(o=lab.split.map(ORDER)).sort_values(["o", "study_uuid"]).reset_index(drop=True)
        rows.append(pd.DataFrame(dict(
            study_id=dataset + "/" + lab.study_uuid, task=task, split=lab.split, label=lab.y, mode=mode,
            cluster_id=dataset + "/" + studies.cluster_id.reindex(lab.study_uuid).values)))
        n = lab.split.value_counts().reindex(["TRAIN", "VAL", "TEST"]).fillna(0).astype(int).to_dict()
        print(f"{task} ({dataset}, {mode}): {len(lab)} studies {n}")

    probe = pd.concat(rows, ignore_index=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    probe.to_csv(out / "cardiobench_probe.csv", index=False)
    zs = probe[(probe["mode"] == "classification") & (probe.split == "TEST")]
    zs[["study_id", "task", "label", "cluster_id"]].assign(label=lambda d: d.label.astype(int)).to_csv(
        out / "cardiobench.csv", index=False)
    print(f"{len(probe)} rows -> {out / 'cardiobench_probe.csv'}; {len(zs)} TEST rows -> {out / 'cardiobench.csv'}")


if __name__ == "__main__":
    main()
