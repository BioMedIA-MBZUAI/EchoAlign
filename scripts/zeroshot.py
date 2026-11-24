#!/usr/bin/env python3
"""Score every task of one cohort with one zero-shot method and write a score CSV.

    python scripts/zeroshot.py --method panecho --cohort mimic_iv_echo \
        --embeddings embeddings/mimic_iv_echo/panecho.npz --out results/mimic_iv_echo/zeroshot/panecho_own.csv

    python scripts/zeroshot.py --method echoprime --cohort mimic_iv_echo --echoprime-repo models/EchoPrime \
        --embeddings embeddings/mimic_iv_echo/echoprime.npz --out results/mimic_iv_echo/zeroshot/echoprime_own.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import iter_npz, load_tasks  # noqa: E402
from eval.zeroshot import METHODS, build  # noqa: E402



def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--cohort", required=True, help="cohort name in the task configuration")
    ap.add_argument("--embeddings", required=True, help=".npz embedding file (see eval/analysis/data.py)")
    ap.add_argument("--out", required=True, help="score file")
    ap.add_argument("--tasks-config", default=str(ROOT / "configs/zeroshot_tasks.yaml"), help="task configuration (YAML)")
    ap.add_argument("--weights", default=str(ROOT / "weights/EchoAlign.pt"), help="EchoAlign weights")
    ap.add_argument("--prompts", default=str(ROOT / "configs/echoalign_prompts.yaml"), help="EchoAlign prompt file (YAML)")
    ap.add_argument("--echoprime-repo", default=str(ROOT / "models/EchoPrime"), help="clone of EchoPrime with model_data")
    ap.add_argument("--echoclip-repo", default=str(ROOT / "models/echo_CLIP"), help="clone of echo_CLIP")
    ap.add_argument("--panecho-source", default=str(ROOT / "models/PanEcho"), help="PanEcho GitHub repository or local clone")
    ap.add_argument("--echoprime-pooling", choices=["sections", "clips"], default="sections",
                    help="echoprime_prompts: score the task's EchoPrime section embedding per study, or each clip")
    ap.add_argument("--echoprime-section-map", default=str(ROOT / "configs/echoprime_sections.yaml"),
                    help="echoprime_prompts: task -> EchoPrime section (YAML)")
    ap.add_argument("--echoprime-section-order", choices=["as_released", "by_name"], default="as_released",
                    help="echoprime: section lookup in predict_metrics (eval/zeroshot/echoprime.py)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", nargs="+", default=None, help="tasks to score (default: all of the cohort)")
    a = ap.parse_args()
    return ap, a


def score_one(ap, a):
    kwargs = {
        "echoalign": lambda: dict(weights=a.weights, prompts=a.prompts, device=a.device),
        "echoprime": lambda: dict(repo_dir=a.echoprime_repo, section_order=a.echoprime_section_order),
        "panecho": lambda: dict(source=a.panecho_source, local=True),
        "echoclip": lambda: dict(repo_dir=a.echoclip_repo, device=a.device),
        "echoprime_prompts": lambda: dict(repo_dir=a.echoprime_repo, prompts=a.prompts, pooling=a.echoprime_pooling,
                                          section_map=a.echoprime_section_map, cohort=a.cohort),
        "echoclip_prompts": lambda: dict(prompts=a.prompts),
    }[a.method]()
    required = {"echoalign": ("prompts", "--prompts"), "echoprime": ("repo_dir", "--echoprime-repo"),
                "echoclip": ("repo_dir", "--echoclip-repo")}.get(a.method)
    if required and not kwargs[required[0]]:
        ap.error(f"--method {a.method} needs {required[1]}")

    tasks = load_tasks(a.tasks_config, a.cohort)
    if a.tasks:
        unknown = set(a.tasks) - set(tasks)
        if unknown:
            ap.error(f"tasks not in cohort {a.cohort}: {sorted(unknown)}")
        tasks = {t: tasks[t] for t in a.tasks}
    scorer = build(a.method, **kwargs)
    scores = pd.concat([scorer.score(e, tasks) for e in iter_npz(a.embeddings)], ignore_index=True)
    scores.insert(2, "method", a.method)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(a.out, index=False)
    n_missing = int(scores.score.isna().sum())
    print(f"{a.method} / {a.cohort}: {scores.task.nunique()} tasks, {len(scores)} scores, {n_missing} missing -> {a.out}")


def main():
    ap, a = parse_args()
    score_one(ap, a)


if __name__ == "__main__":
    main()
