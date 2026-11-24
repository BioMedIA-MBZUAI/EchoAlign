import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval.analysis.data import study_mean


def load_echoclip_repo(repo_dir):
    repo = Path(repo_dir).resolve()
    spec = importlib.util.spec_from_file_location("echoclip_utils", repo / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    return utils, json.loads((repo / "prompts_used.json").read_text())


class EchoCLIPPrompts:
    name = "echoclip"

    def __init__(self, repo_dir, model_id="hf-hub:mkaichristensen/echo-clip", device=None):
        import open_clip
        self.utils, self.prompts = load_echoclip_repo(repo_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(model_id, precision="fp32", device=self.device)
        self.model.eval()
        self.tokenize = open_clip.tokenize

    @torch.inference_mode()
    def encode_text(self, texts):
        e = self.model.encode_text(self.tokenize(list(texts)).to(self.device)).float()
        return torch.nn.functional.normalize(e, dim=-1).cpu()

    def prompt_set(self, key, rule):
        d = self.prompts[key]
        if rule.startswith("grades:"):
            keep = {int(g) for g in rule[7:].split(",")}
            return [t for t, v in zip(d["label_sources"], d["values"]) if int(v) in keep]
        return list(d["label_sources"])

    def clip_scores(self, video, key, rule):
        if rule in ("reg", "negreg"):
            lo, hi = self.prompts[key]["range"]
            texts, values = [], []
            for t in self.prompts[key]["label_sources"]:
                for i in range(lo, hi + 1):
                    texts.append(t.replace("<#>", str(i)))
                    values.append(i)
            v = self.utils.compute_regression_metric(video, self.encode_text(texts), values).numpy()
            return -v if rule == "negreg" else v
        if rule != "binary" and not rule.startswith("grades:"):
            raise ValueError(f"unknown EchoCLIP rule {rule!r}")
        return self.utils.compute_binary_metric(video, self.encode_text(self.prompt_set(key, rule))).numpy()

    def score(self, emb, tasks):
        frames = torch.as_tensor(emb["frame_features"], dtype=torch.float32)
        n_frames = np.asarray(emb["n_frames"])
        out = []
        for task, spec in tasks.items():
            s = spec.get(self.name)
            if not s:
                continue
            per_clip = np.empty(len(frames))
            for n in np.unique(n_frames):                 # EchoCLIP's functions average over all frames given
                idx = np.flatnonzero(n_frames == n)
                per_clip[idx] = self.clip_scores(frames[idx, :n], s["output"], s["rule"])
            m = study_mean(per_clip, emb["study_id"])[0]
            out.append(pd.DataFrame(dict(study_id=m.index, task=task, score=np.asarray(m))))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["study_id", "task", "score"])
