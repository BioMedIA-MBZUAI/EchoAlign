import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SHARD_NAME = re.compile(r"\.shard-(\d{3})-of-(\d{3})\.npz$")
ROOT = Path(__file__).resolve().parents[2]
COHORTS = ROOT / "configs/cohorts.yaml"
COHORT_PATHS = ("manifest", "zeroshot_labels", "probe_labels")


def shard_path(path, shard, num_shards):
    p = Path(path)
    return p.with_name(f"{p.stem}.shard-{shard:03d}-of-{num_shards:03d}.npz")


def embedding_files(path):
    p = Path(path)
    if p.exists():
        if any(p.parent.glob(f"{p.stem}.shard-*-of-*.npz")):
            print(f"warning: reading {p}; its shard files are ignored", flush=True)
        return [p]
    files = sorted(p.parent.glob(f"{p.stem}.shard-*-of-*.npz"))
    if not files:
        raise FileNotFoundError(f"{p} does not exist and has no shard files")
    totals = {int(SHARD_NAME.search(f.name).group(2)) for f in files}
    if len(totals) != 1 or len(files) != totals.pop():
        raise ValueError(f"incomplete or mixed shard set for {p}: {[f.name for f in files]}")
    return files


def _read(path):
    with np.load(path, allow_pickle=False) as f:
        out = {k: f[k] for k in f.files}
    for k in ("clip_id", "study_id"):
        if k in out:
            out[k] = out[k].astype(str)
    return out


def iter_npz(path):
    for f in embedding_files(path):
        yield _read(f)


def load_npz(path):
    parts = list(iter_npz(path))
    if len(parts) == 1:
        return parts[0]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def load_labels(path):
    d = pd.read_csv(path, dtype={"study_id": str, "task": str, "cluster_id": str, "split": str})
    if "split" in d:
        d = d[d.split == "TEST"]
        binary = d.groupby("task").label.transform(lambda v: v.isin([0, 1]).all())
        if "mode" in d:
            binary &= d["mode"].fillna("classification").eq("classification")
        d = d[binary]
    missing = {"study_id", "task", "label"} - set(d.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "cluster_id" not in d:
        d["cluster_id"] = d["study_id"]
    if not d["label"].isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")
    if d.duplicated(["study_id", "task"]).any():
        raise ValueError(f"{path}: duplicate study_id and task rows")
    return d


def load_tasks(path, cohort):
    cfg = yaml.safe_load(open(path))
    if cohort not in cfg:
        raise KeyError(f"cohort {cohort!r} not in {path}; choose from {sorted(cfg)}")
    return cfg[cohort]


def datasets_config():
    local = ROOT / "configs/datasets.local.yaml"
    return local if local.exists() else ROOT / "configs/datasets.yaml"


def load_cohort(name, path=COHORTS):
    cfg = yaml.safe_load(open(path))
    if name not in cfg:
        raise KeyError(f"cohort {name!r} not in {path}; choose from {sorted(cfg)}")
    return {k: str(ROOT / v) if k in COHORT_PATHS else v for k, v in cfg[name].items()}


def study_mean(values, study_ids):
    v = np.asarray(values, float)
    frame = pd.DataFrame(v.reshape(len(v), -1))
    frame["study_id"] = np.asarray(study_ids, str)
    return frame.groupby("study_id", sort=True).mean()
