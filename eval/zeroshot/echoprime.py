import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval.analysis.echoprime_sections import N_VIEWS


@contextlib.contextmanager
def inside(repo):
    """Run inside the EchoPrime clone: its modules read assets/ relative to the working directory."""
    old = os.getcwd()
    os.chdir(repo)
    try:
        yield
    finally:
        os.chdir(old)


def import_echoprime(repo_dir):
    repo = Path(repo_dir).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    with inside(repo):
        import utils
        from echo_prime.model import EchoPrime
    return repo, utils, EchoPrime


def load_echoprime(repo_dir):
    repo, _, EchoPrime = import_echoprime(repo_dir)

    ep = EchoPrime.__new__(EchoPrime)
    cand = repo / "model_data" / "candidates_data"
    ep.MIL_weights = pd.read_csv(repo / "assets" / "MIL_weights.csv")
    ep.non_empty_sections = ep.MIL_weights["Section"]
    ep.section_weights = ep.MIL_weights.iloc[:, 1:].to_numpy()
    ep.candidate_studies = list(pd.read_csv(cand / "candidate_studies.csv")["Study"])
    ep.candidate_embeddings = torch.cat([torch.load(cand / f"candidate_embeddings_p{i}.pt") for i in (1, 2)], dim=0)
    ep.candidate_labels = pd.read_pickle(cand / "candidate_labels.pkl")
    ep.section_to_phenotypes = pd.read_pickle(repo / "assets" / "section_to_phenotypes.pkl")
    return ep


class EchoPrimeRetrieval:
    name = "echoprime"

    def __init__(self, repo_dir, k=50, section_order="as_released"):
        """section_order "as_released" matches sections by position (10 of 11 groups read another section); "by_name" uses MIL_weights.csv order."""
        if section_order not in ("as_released", "by_name"):
            raise ValueError(f"unknown section_order {section_order!r}")
        self.ep = load_echoprime(repo_dir)
        self.k = k
        self.section_order = section_order
        if section_order == "by_name":
            groups = dict(self.ep.section_to_phenotypes)
            names = [str(s) for s in self.ep.non_empty_sections]
            missing = set(map(str, groups)) - set(names)
            if missing:
                raise ValueError(f"sections without a MIL_weights.csv row: {sorted(missing)}")
            self.ep.section_to_phenotypes = {n: groups.get(n, []) for n in names}

    def phenotypes(self, emb):
        feats = torch.as_tensor(emb["features"], dtype=torch.float32)
        views = torch.nn.functional.one_hot(torch.as_tensor(emb["views"], dtype=torch.long), N_VIEWS).float()
        study = pd.Series(emb["study_id"])
        rows = {}
        for sid, idx in study.groupby(study).groups.items():
            idx = np.asarray(idx)
            study_embedding = torch.cat([feats[idx], views[idx]], dim=1)
            rows[sid] = self.ep.predict_metrics(study_embedding, k=self.k)
        return pd.DataFrame.from_dict(rows, orient="index").sort_index()

    def score(self, emb, tasks):
        table = self.phenotypes(emb)
        out = []
        for task, spec in tasks.items():
            s = spec.get(self.name)
            if not s:
                continue
            v = table[s["output"]]
            if s["rule"] == "neg":
                v = -v
            elif s["rule"] != "label":
                raise ValueError(f"{task}: unknown EchoPrime rule {s['rule']!r}")
            out.append(pd.DataFrame(dict(study_id=v.index, task=task, score=v.to_numpy())))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["study_id", "task", "score"])
