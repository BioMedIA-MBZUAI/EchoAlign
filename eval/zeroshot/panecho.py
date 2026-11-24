import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval.analysis.data import study_mean


def hub_load(source="models/PanEcho", local=True, **kwargs):
    import timm  # noqa: F401
    return torch.hub.load(source, "PanEcho", source="local" if local else "github", pretrained=True,
                          trust_repo=True, **kwargs)


def load_panecho(source="models/PanEcho", local=True):
    model = hub_load(source, local, tasks="all")
    model.encoder = torch.nn.Identity()
    return model.eval()


class PanEchoHeads:
    name = "panecho"

    def __init__(self, source="models/PanEcho", local=True, batch=4096):
        self.model = load_panecho(source, local)
        self.classes = {t.task_name: [str(c) for c in t.class_names] for t in self.model.tasks}
        self.batch = batch

    @torch.inference_mode()
    def head_outputs(self, features):
        x = torch.as_tensor(features, dtype=torch.float32)
        parts = [self.model(x[i:i + self.batch]) for i in range(0, len(x), self.batch)]
        return {k: torch.cat([p[k] for p in parts]).reshape(len(x), -1).numpy() for k in parts[0]}

    def score(self, emb, tasks):
        heads = self.head_outputs(emb["features"])
        out = []
        for task, spec in tasks.items():
            s = spec.get(self.name)
            if not s:
                continue
            head, rule = s["output"], s["rule"]
            m = study_mean(heads[head], emb["study_id"])
            cls = self.classes[head]
            if rule in ("binary", "reg"):
                v = m[0]
            elif rule == "expgrade":
                grade = {"None": 0, "Mild|Moderate": 1, "Severe": 2}
                v = sum(m[i] * grade[c] for i, c in enumerate(cls))
            elif rule.startswith("pos:"):
                v = m[[cls.index(c) for c in rule[4:].split(";")]].sum(axis=1)
            else:
                raise ValueError(f"{task}: unknown PanEcho rule {rule!r}")
            out.append(pd.DataFrame(dict(study_id=m.index, task=task, score=np.asarray(v))))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["study_id", "task", "score"])
