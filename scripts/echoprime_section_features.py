#!/usr/bin/env python3
"""Build EchoPrime's per-section study embeddings from clip embeddings.

    python scripts/echoprime_section_features.py --embeddings embeddings/mimic_iv_echo/echoprime.npz \
        --echoprime-repo models/EchoPrime --out embeddings/mimic_iv_echo/echoprime_sections.npz
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import load_npz  # noqa: E402
from eval.analysis.echoprime_sections import load_mil_weights, section_vectors, study_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True, help=".npz from scripts/embed.py")
    ap.add_argument("--echoprime-repo", default=str(ROOT / "models/EchoPrime"), help="clone of EchoPrime (assets/MIL_weights.csv)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    emb = load_npz(a.embeddings)
    feats, views, study = np.asarray(emb["features"], np.float32), np.asarray(emb["views"]), emb["study_id"]
    info = dict(cache_rows=int(len(study)), rows_with_study=int(len(study)), rows_dropped=0)
    names, weights = load_mil_weights(a.echoprime_repo)
    studies, index = study_index(study)
    vectors, total = section_vectors(feats, views, index, len(studies), weights)
    n_clips = np.bincount(index, minlength=len(studies))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, study_id=studies, sections=vectors, section_names=np.array(names), total_weight=total,
             n_clips=n_clips)
    info.update(studies=int(len(studies)), clips=int(len(index)), view_counts=np.bincount(views, minlength=11).tolist(),
                zero_weight_study_sections={n: int((total[:, k] == 0).sum()) for k, n in enumerate(names)},
                seconds=round(time.time() - t0, 1))
    out.with_suffix(".json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
