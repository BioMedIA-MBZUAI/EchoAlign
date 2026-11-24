from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml

N_VIEWS = 11


def load_mil_weights(repo_dir):
    table = pd.read_csv(Path(repo_dir) / "assets" / "MIL_weights.csv")
    weights = table.iloc[:, 1:].to_numpy(np.float64)
    if weights.shape[1] != N_VIEWS:
        raise ValueError(f"expected {N_VIEWS} view columns in MIL_weights.csv, found {weights.shape[1]}")
    return list(table["Section"]), weights


def study_index(study_ids):
    studies, index = np.unique(np.asarray(study_ids, str), return_inverse=True)
    return studies, index


def _membership(index, n_studies, values=None):
    n = len(index)
    data = np.ones(n, np.float64) if values is None else np.asarray(values, np.float64)
    return sp.csr_matrix((data, (np.asarray(index), np.arange(n))), shape=(n_studies, n))


def l2_normalize(x, eps=1e-12):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def section_vectors(features, views, index, n_studies, weights):
    feats = np.asarray(features)
    views = np.asarray(views)
    out = np.zeros((n_studies, len(weights), feats.shape[1]), np.float32)
    total = np.zeros((n_studies, len(weights)), np.float64)
    for k, row in enumerate(np.asarray(weights, np.float64)):
        w = row[views]
        a = _membership(index, n_studies, w)
        out[:, k] = l2_normalize(a @ feats).astype(np.float32)
        total[:, k] = np.asarray(a.sum(1)).ravel()
    return out, total


def load_section_map(path, cohort):
    cfg = yaml.safe_load(open(path))
    if cohort not in cfg:
        raise KeyError(f"cohort {cohort!r} not in {path}; choose from {sorted(k for k in cfg if k != 'sections')}")
    return cfg[cohort]


def check_section_map(section_map, sections):
    names = set(sections)
    for task, spec in section_map.items():
        used = [spec["section"]] if spec.get("section") else []
        bad = [s for s in used if s not in names]
        if bad:
            raise ValueError(f"{task}: unknown EchoPrime section(s) {bad}")


def task_vectors(spec, sections, vectors):
    if spec.get("section"):
        return vectors[:, sections.index(spec["section"])], f"section:{spec['section']}"
    return l2_normalize(vectors.astype(np.float64).mean(1)).astype(np.float32), "fallback:all_sections"


def section_map_table(section_map):
    rows = []
    for task, spec in section_map.items():
        rows.append(dict(task=task, mode=spec.get("mode", "classification"), section=spec.get("section") or "",
                         source=spec["source"], echoprime_phenotype=spec.get("phenotype") or "",
                         note=spec.get("note") or ""))
    return pd.DataFrame(rows)
