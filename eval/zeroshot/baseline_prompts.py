from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from eval.analysis.data import study_mean
from eval.analysis.echoprime_sections import (check_section_map, l2_normalize, load_mil_weights, load_section_map,
                                              section_vectors, study_index, task_vectors)


class _BaselinePrompts:
    name = None
    native = None

    def __init__(self, prompts):
        self.prompts = yaml.safe_load(open(prompts)) if isinstance(prompts, (str, Path)) else prompts

    def score(self, emb, tasks):
        out = []
        for task, spec in tasks.items():
            if spec.get(self.native) or task not in self.prompts:
                continue
            pos = l2_normalize(self.encode_text(self.prompts[task]["positive"]))
            neg = l2_normalize(self.encode_text(self.prompts[task]["negative"]))
            m = study_mean(self.clip_scores(emb, pos, neg), emb["study_id"])[0]
            out.append(pd.DataFrame(dict(study_id=m.index, task=task, score=np.asarray(m))))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["study_id", "task", "score"])


class EchoPrimePrompts(_BaselinePrompts):
    name, native = "echoprime_prompts", "echoprime"

    def __init__(self, repo_dir, prompts, pooling="sections", section_map=None, cohort=None):
        super().__init__(prompts)
        from .echoprime import import_echoprime, inside
        repo, _, _ = import_echoprime(repo_dir)
        with inside(repo):
            from echo_prime.model import EchoPrimeTextEncoder
            self.te = EchoPrimeTextEncoder(device="cpu")
        self.te.load_state_dict(torch.load(repo / "model_data/weights/echo_prime_text_encoder.pt", map_location="cpu"),
                                strict=True)
        self.te.eval()
        if pooling not in ("sections", "clips"):
            raise ValueError(f"unknown pooling {pooling!r}; choose sections or clips")
        self.pooling = pooling
        if pooling == "sections":
            if not section_map or not cohort:
                raise ValueError("pooling='sections' needs section_map and cohort")
            self.section_names, self.mil_weights = load_mil_weights(repo)
            self.section_map = load_section_map(section_map, cohort)
            check_section_map(self.section_map, self.section_names)

    def score(self, emb, tasks):
        if self.pooling == "clips":
            return super().score(emb, tasks)
        studies, index = study_index(emb["study_id"])
        feats = np.asarray(emb["features"], np.float32)
        vectors, _ = section_vectors(feats, emb["views"], index, len(studies), self.mil_weights)
        out = []
        for task, spec in tasks.items():
            if spec.get(self.native) or task not in self.prompts:
                continue
            if task not in self.section_map:
                raise KeyError(f"{task} has no entry in the EchoPrime section map")
            pos = l2_normalize(self.encode_text(self.prompts[task]["positive"]))
            neg = l2_normalize(self.encode_text(self.prompts[task]["negative"]))
            v, _ = task_vectors(self.section_map[task], self.section_names, vectors)
            v = l2_normalize(v)
            out.append(pd.DataFrame(dict(study_id=studies, task=task, score=(v @ pos.T).mean(1) - (v @ neg.T).mean(1))))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["study_id", "task", "score"])

    @torch.inference_mode()
    def encode_text(self, texts):
        return torch.cat([self.te(t) for t in texts]).float().numpy()   # one text per call, as in EchoPrime

    def clip_scores(self, emb, pos, neg):
        x = l2_normalize(emb["features"])
        return (x @ pos.T).mean(1) - (x @ neg.T).mean(1)


class EchoCLIPPromptsAnyTask(_BaselinePrompts):
    name, native = "echoclip_prompts", "echoclip"

    def __init__(self, prompts, model_id="hf-hub:mkaichristensen/echo-clip"):
        super().__init__(prompts)
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms(model_id, precision="fp32", device="cpu")
        self.model.eval()
        self.tokenize = open_clip.tokenize

    @torch.inference_mode()
    def encode_text(self, texts):
        return self.model.encode_text(self.tokenize(list(texts))).float().numpy()

    def clip_scores(self, emb, pos, neg):
        f, n = np.asarray(emb["frame_features"], np.float64), np.asarray(emb["n_frames"])
        mask = (np.arange(f.shape[1])[None] < n[:, None]).astype(np.float64)
        per_frame = (f @ pos.T).mean(-1) - (f @ neg.T).mean(-1)
        return (per_frame * mask).sum(1) / np.maximum(n, 1)
