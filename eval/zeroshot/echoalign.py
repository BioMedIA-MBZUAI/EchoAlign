from pathlib import Path

import pandas as pd
import yaml

from eval.zeroshot.prompt_selection import study_matrix


class EchoAlignPrompts:
    name = "echoalign"

    def __init__(self, weights, prompts, device=None):
        from echoalign.model import EchoAlign
        self.model = EchoAlign(weights, device=device)
        self.prompts = yaml.safe_load(open(prompts)) if isinstance(prompts, (str, Path)) else prompts

    def score(self, emb, tasks):
        out = []
        for task, spec in tasks.items():
            if task not in self.prompts:
                continue
            output = self.prompts[task].get("output") or (spec.get(self.name) or {}).get("structure")
            x = study_matrix(emb["vectors"], output)
            pos = self.model.encode_text(self.prompts[task]["positive"]).astype(float)
            neg = self.model.encode_text(self.prompts[task]["negative"]).astype(float)
            s = (x @ pos.T).mean(1) - (x @ neg.T).mean(1)
            out.append(pd.DataFrame(dict(study_id=emb["study_id"], task=task, score=s)))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["study_id", "task", "score"])
