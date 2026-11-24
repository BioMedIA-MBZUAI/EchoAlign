#!/usr/bin/env python3
"""Select EchoAlign's zero-shot prompts on the private validation split and write the prompt file.

    python scripts/select_prompts.py --weights weights/EchoAlign.pt \
        --embeddings embeddings/private/echoalign.npz \
        --out configs/echoalign_prompts.yaml --report results/prompt_selection.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import iter_npz, load_cohort, load_tasks  # noqa: E402
from eval.zeroshot.prompt_selection import (CAP, MIN_GAIN, cardiobench_prompts, fill_cohorts,  # noqa: E402
                                            output_order, select, study_matrix)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="EchoAlign weights")
    ap.add_argument("--embeddings", required=True, help="private study embeddings of the same weights (.npz)")
    ap.add_argument("--labels", default=None, help="private probe labels (CSV; default: configs/cohorts.yaml)")
    ap.add_argument("--candidates", default=str(ROOT / "configs/prompt_candidates.yaml"))
    ap.add_argument("--tasks-config", default=str(ROOT / "configs/zeroshot_tasks.yaml"))
    ap.add_argument("--cardiobench-prompts", default=str(ROOT / "benchmarks/CardioBench/src/prompts.py"))
    ap.add_argument("--out", required=True, help="prompt file to write (YAML)")
    ap.add_argument("--report", required=True, help="selection report to write (JSON)")
    ap.add_argument("--cap", type=int, default=CAP, help="most sentences per class")
    ap.add_argument("--min-gain", type=float, default=MIN_GAIN, help="least VAL AUROC gain to add a sentence")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def val_labels(path, tasks):
    d = pd.read_csv(path, dtype={"study_id": str, "task": str, "split": str})
    d = d[d.split.eq("VAL") & d.task.isin(tasks)]
    if "mode" in d:
        d = d[d["mode"].fillna("classification").eq("classification")]
    bad = sorted(d.task[~d.label.isin([0, 1])].unique())
    if bad:
        raise ValueError(f"{path}: VAL labels of {bad} must be 0 or 1")
    return d


def val_vectors(path, study_ids):
    keep = set(study_ids)
    ids, vec = [], []
    for e in iter_npz(path):
        m = np.isin(e["study_id"], list(keep))
        ids.append(e["study_id"][m])
        vec.append(np.asarray(e["vectors"][m], np.float32))
    return pd.Index(np.concatenate(ids)), np.concatenate(vec)


def main():
    a = parse_args()
    from echoalign.model import EchoAlign

    cand = yaml.safe_load(open(a.candidates))
    tasks = load_tasks(a.tasks_config, "private")
    missing = sorted(set(tasks) - set(cand["findings"]))
    if missing:
        raise SystemExit(f"private tasks without candidates: {missing}")
    a.labels = a.labels or load_cohort("private")["probe_labels"]
    labels = val_labels(a.labels, set(tasks))
    ids, vectors = val_vectors(a.embeddings, labels.study_id.unique())
    if ids.duplicated().any():
        raise SystemExit(f"{a.embeddings}: duplicate study ids")

    model = EchoAlign(a.weights, device=a.device)
    texts = sorted({t for c in cand["findings"].values() for cls in ("positive", "negative") for t in c[cls]})
    text_emb = dict(zip(texts, np.concatenate([model.encode_text(texts[i:i + 64])
                                               for i in range(0, len(texts), 64)]).astype(np.float64)))

    selected, report = {}, {}
    for task, spec in tasks.items():
        configured = (spec.get("echoalign") or {}).get("structure")
        lab = labels[labels.task.eq(task)]
        row = ids.get_indexer(lab.study_id)
        have = row >= 0
        y = lab.label.to_numpy(int)[have]
        if len(set(y)) < 2:
            raise SystemExit(f"{task}: VAL labels hold one class")
        pos_txt, neg_txt = cand["findings"][task]["positive"], cand["findings"][task]["negative"]
        pos_emb, neg_emb = np.stack([text_emb[t] for t in pos_txt]), np.stack([text_emb[t] for t in neg_txt])
        best, by_output = None, {}
        for output in output_order(configured):
            x = study_matrix(vectors[row[have]], output)
            p, n, auroc, trace = select(x @ pos_emb.T, x @ neg_emb.T, y, a.cap, a.min_gain)
            by_output[output] = auroc
            if output == (configured or "global"):  # the other outputs are only reported
                best = (output, p, n, auroc, trace)
        output, p, n, auroc, trace = best
        selected[task] = dict(output=output, positive=[pos_txt[i] for i in p], negative=[neg_txt[j] for j in n])
        report[task] = dict(configured_output=configured or "global", n_val_labels=int(len(lab)), n_val=int(have.sum()),
                            n_val_positive=int(y.sum()), n_candidates=dict(positive=len(pos_txt), negative=len(neg_txt)),
                            val_auroc=auroc, **selected[task], val_auroc_by_output=by_output, trace=trace)
        print(f"{task:40s} {output:16s} (configured {str(configured or 'global'):16s} "
              f"{by_output[configured or 'global']:.4f}) VAL n={have.sum()} pos={y.sum()} AUROC={auroc:.4f} "
              f"sentences {len(p)}+{len(n)} (missing embeddings: {(~have).sum()})", flush=True)

    prompts = fill_cohorts(selected, cand, cardiobench_prompts(a.cardiobench_prompts), a.cap)
    for cohort in ("mimic_iv_echo", "cardiobench"):
        absent = sorted(set(load_tasks(a.tasks_config, cohort)) - set(prompts))
        if absent:
            raise SystemExit(f"{cohort} tasks without prompts: {absent}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"# EchoAlign zero-shot prompts, one entry per task. Written by scripts/select_prompts.py "
                f"(up to {a.cap} sentences per class, selected on private VAL for the output of the finding's structure).\n")
        yaml.safe_dump(prompts, f, sort_keys=False, allow_unicode=True, width=1000)
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    meta = dict(weights=str(a.weights), embeddings=str(a.embeddings), labels=str(a.labels), split="VAL",
                candidates=str(a.candidates), cap=a.cap, min_gain=a.min_gain)
    Path(a.report).write_text(json.dumps(dict(meta, findings=report), indent=1, ensure_ascii=False) + "\n")
    print(f"{len(selected)} findings selected, {len(prompts)} prompt entries -> {a.out}; report -> {a.report}")


if __name__ == "__main__":
    main()
