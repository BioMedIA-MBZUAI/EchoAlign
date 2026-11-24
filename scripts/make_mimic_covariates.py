#!/usr/bin/env python3
"""Build the per-study MIMIC-IV-Echo covariate table read by scripts/subgroups.py --covariates.

    python scripts/make_mimic_covariates.py --mimic-echo <MIMIC-IV-Echo folder> --mimic-hosp <MIMIC-IV hosp folder>
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pydicom

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ["study_id", "subject_id", "study_datetime", "measurement_id", "gender", "anchor_age", "anchor_year", "age",
           "technical_quality", "scanner_manufacturer", "scanner_model", "bmi", "resting_sbp", "resting_dbp"]


def first_file(folder, name):
    for f in (folder / f"{name}.gz", folder / name):
        if f.exists():
            return f
    raise SystemExit(f"no {name}(.gz) in {folder}")


def tags(path):
    d = pydicom.dcmread(path, stop_before_pixels=True)
    return str(d.get("Manufacturer", "")), str(d.get("ManufacturerModelName", ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mimic-echo", required=True, help="root of the MIMIC-IV-Echo v1.0 download (holds meta/ and files/)")
    ap.add_argument("--mimic-hosp", required=True, help="MIMIC-IV hosp folder (holds patients.csv.gz)")
    ap.add_argument("--split", default=str(ROOT / "data/splits/mimic_iv_echo.csv"), help="from scripts/make_mimic_labels.py")
    ap.add_argument("--splits", nargs="+", default=["TEST"], choices=["TRAIN", "VAL", "TEST"])
    ap.add_argument("--out", default=str(ROOT / "data/labels/mimic_iv_echo_covariates.csv"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    root = Path(a.mimic_echo)
    meta = root / "meta" if (root / "meta").is_dir() else root
    split = pd.read_csv(a.split, dtype=str)
    test = set(split.study_id[split.split.isin(a.splits)])

    d = pd.read_csv(first_file(meta, "echo-study-list.csv"), dtype={"study_id": str, "subject_id": str, "measurement_id": str})
    d = d[d.study_id.isin(test)]
    assert d.study_id.is_unique
    pt = pd.read_csv(first_file(Path(a.mimic_hosp), "patients.csv"), dtype={"subject_id": str})
    d = d.merge(pt, on="subject_id", how="left")
    d["age"] = d.anchor_age + (pd.to_datetime(d.study_datetime).dt.year - d.anchor_year)

    sm = pd.read_csv(first_file(meta, "structured-measurement.csv"), dtype={"measurement_id": str},
                     usecols=["measurement_id", "measurement", "result"])
    tq = sm[sm.measurement == "technical_quality"].dropna(subset=["result"]).drop_duplicates("measurement_id")
    d["technical_quality"] = d.measurement_id.map(tq.set_index("measurement_id").result)

    def meas(name):
        x = sm[sm.measurement == name].assign(v=lambda t: pd.to_numeric(t.result, errors="coerce")).dropna(subset=["v"])
        return d.measurement_id.map(x.groupby("measurement_id").v.first())

    h = meas("height_cm").fillna(meas("height_inches") * 2.54)
    w = meas("weight_kg").fillna(meas("weight_pounds") * 0.45359237)
    d["bmi"] = (w / (h / 100) ** 2).where(h.between(122, 213) & w.between(32, 272))
    sbp, dbp = meas("resting_sbp"), meas("resting_dbp")
    ok = sbp.between(60, 260) & dbp.between(30, 160)
    d["resting_sbp"], d["resting_dbp"] = sbp.where(ok), dbp.where(ok)

    rec = pd.read_csv(first_file(meta, "echo-record-list.csv"), dtype={"study_id": str})
    rec = rec[rec.study_id.isin(test)].sort_values(["study_id", "dicom_filepath"])
    g = rec.groupby("study_id")
    pick = pd.concat([g.head(1), g.tail(1)]).drop_duplicates()
    with ThreadPoolExecutor(a.workers) as ex:
        pick[["scanner_manufacturer", "scanner_model"]] = list(ex.map(tags, [root / p for p in pick.dicom_filepath]))
    assert pick.groupby("study_id").scanner_model.nunique().max() == 1
    sc = pick.drop_duplicates("study_id").set_index("study_id").replace("", None)
    d["scanner_manufacturer"] = d.study_id.map(sc.scanner_manufacturer)
    d["scanner_model"] = d.study_id.map(sc.scanner_model)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d[COLUMNS].to_csv(out, index=False, lineterminator="\n")
    print(f"wrote {out}: {len(d):,} of {len(test):,} studies in {' '.join(a.splits)}")


if __name__ == "__main__":
    main()
