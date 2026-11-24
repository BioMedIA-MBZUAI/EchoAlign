"""Prepare the inputs of scripts/compare_paired.py for one cohort.

    python scripts/compare_inputs.py prepare --labels data/labels/mimic_iv_echo_probe.csv --reference echoalign \
        --method echoalign=results/mimic_iv_echo/probe/echoalign_test_scores.csv \
        --method echoprime=results/mimic_iv_echo/probe/echoprime_test_scores.csv \
        --out results/mimic_iv_echo/compare_test_scores

    python scripts/compare_inputs.py subset --scores results/mimic_iv_echo/zeroshot/echoprime_prompts_all.csv \
        --exclude-tasks-of results/mimic_iv_echo/zeroshot/echoprime_own.csv \
        --out results/mimic_iv_echo/zeroshot/echoprime_prompts.csv
"""
import argparse
from pathlib import Path

import pandas as pd

def read_scores(spec):
    name, files = spec.split("=", 1)
    s = pd.concat([pd.read_csv(f, dtype={"study_id": str}) for f in files.split(",")], ignore_index=True)
    if "score" not in s:
        s = s.rename(columns={next(c for c in ("y_pred", "prediction") if c in s): "score"})
    dup = s.duplicated(["study_id", "task"])
    if dup.any():
        raise SystemExit(f"{name}: {int(dup.sum())} repeated study and task rows")
    return name, s


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = dict(read_scores(m) for m in args.method)

    lab = pd.read_csv(args.labels, dtype={"study_id": str, "cluster_id": str})
    if "split" in lab:
        lab = lab[lab.split == "TEST"]
    else:
        lab = lab.assign(split="TEST")
    unit = "cluster_id" if "cluster_id" in lab else "study_id"
    pd.DataFrame({unit: sorted(lab[unit].unique())}).to_csv(out / "universe.csv", index=False)

    labeled = lab.groupby("task").study_id.agg(set)
    for name, s in scores.items():
        has = s[s.score.notna()].groupby("task").study_id.agg(set)
        absent = [t for t in labeled.index if not (has.get(t, set()) & labeled[t])]
        scores[name] = s[~s.task.isin(absent)]
        scores[name].to_csv(out / f"scores_{name}.csv", index=False)

    reference = scores[args.reference] if args.reference else None
    rows = []
    for task, g in lab.groupby("task"):
        if reference is not None and not (reference.task == task).any():
            continue
        scored = [set(s.study_id[(s.task == task) & s.score.notna()]) for s in scores.values()
                  if (s.task == task).any()]
        keep = set.intersection(*scored) if scored else set()
        rows.append(g[g.study_id.isin(keep)])
    if not rows:
        raise SystemExit("no labeled task has scores")
    paired = pd.concat(rows, ignore_index=True)
    paired.to_csv(out / "paired_labels.csv", index=False)
    print(f"{paired.task.nunique()} tasks, {len(paired)} TEST rows kept of {len(lab)}")


def subset(args):
    own = pd.read_csv(args.exclude_tasks_of, dtype={"study_id": str})
    pr = pd.read_csv(args.scores, dtype={"study_id": str})
    pr[~pr.task.isin(set(own.task))].to_csv(args.out, index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    a = sub.add_parser("prepare", help="joined score files, paired labels and the bootstrap universe")
    a.add_argument("--labels", required=True)
    a.add_argument("--method", action="append", required=True, help="NAME=CSV[,CSV...]")
    a.add_argument("--reference", help="keep only the tasks this method reports (default: every labeled task)")
    a.add_argument("--out", required=True)
    c = sub.add_parser("subset", help="score rows of the tasks another score file does not cover")
    c.add_argument("--scores", required=True, help="score CSV to filter (for example <m>_prompts_all.csv)")
    c.add_argument("--exclude-tasks-of", required=True, help="score CSV whose tasks are left out (<m>_own.csv)")
    c.add_argument("--out", required=True)
    args = p.parse_args()
    dict(prepare=prepare, subset=subset)[args.step](args)


if __name__ == "__main__":
    main()
