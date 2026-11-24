import ast

import numpy as np
from sklearn.metrics import roc_auc_score

from echoalign.pooling import STRUCTURES
from eval.analysis.echoprime_sections import l2_normalize as unit

CAP = 3
MIN_GAIN = 0.005
OUTPUTS = ["global", *STRUCTURES, "mean"]


def study_matrix(vectors, output):
    if output == "mean":
        return unit(sum(unit(vectors[:, k]) for k in range(len(OUTPUTS) - 1)) / (len(OUTPUTS) - 1))
    return unit(vectors[:, 0 if output in (None, "global") else list(STRUCTURES).index(output) + 1])


def select(sim_pos, sim_neg, y, cap=CAP, min_gain=MIN_GAIN):
    def auc(p, n):
        return roc_auc_score(y, sim_pos[:, p].mean(1) - sim_neg[:, n].mean(1))

    pairs = [(i, j) for i in range(sim_pos.shape[1]) for j in range(sim_neg.shape[1])]
    scores = [auc([i], [j]) for i, j in pairs]
    k = int(np.argmax(scores))
    pos, neg, best = [pairs[k][0]], [pairs[k][1]], scores[k]
    trace = [dict(step="pair", positive=pos[0], negative=neg[0], auroc=float(best))]
    while True:
        cand = [("positive", i) for i in range(sim_pos.shape[1]) if i not in pos and len(pos) < cap]
        cand += [("negative", j) for j in range(sim_neg.shape[1]) if j not in neg and len(neg) < cap]
        if not cand:
            break
        scores = [auc(pos + [i], neg) if c == "positive" else auc(pos, neg + [i]) for c, i in cand]
        k = int(np.argmax(scores))
        if scores[k] - best < min_gain - 1e-12:
            trace.append(dict(step="stop", best_gain=float(scores[k] - best)))
            break
        (pos if cand[k][0] == "positive" else neg).append(cand[k][1])
        best = scores[k]
        trace.append(dict(step="add", cls=cand[k][0], index=cand[k][1], auroc=float(best)))
    return pos, neg, float(best), trace


def cardiobench_prompts(path):
    for node in ast.parse(open(path).read()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "zero_shot_prompts" for t in node.targets):
            return {k: [s.strip() for s in v] for k, v in ast.literal_eval(node.value).items()}
    raise KeyError(f"zero_shot_prompts not found in {path}")


def output_order(configured):
    first = configured or "global"
    return [first] + [o for o in OUTPUTS if o != first]


def fill_cohorts(selected, candidates, benchmark_prompts, cap=CAP):
    findings = dict(selected)
    for name, c in candidates["findings"].items():
        findings.setdefault(name, {cls: c[cls][:cap] for cls in ("positive", "negative")})
    out = dict(findings)
    for cohort, tasks in candidates.get("reuse", {}).items():
        for task, finding in tasks.items():
            if task in out and out[task] != findings[finding]:
                raise ValueError(f"{cohort}/{task}: name already holds different prompts")
            out[task] = findings[finding]
    for task, keys in candidates.get("benchmark", {}).get("cardiobench", {}).items():
        out[task] = {cls: benchmark_prompts[keys[cls]] for cls in ("positive", "negative")}
    keys = ("output", "positive", "negative")
    return {k: {c: (list(v[c]) if c != "output" else v[c]) for c in keys if c in v} for k, v in sorted(out.items())}
