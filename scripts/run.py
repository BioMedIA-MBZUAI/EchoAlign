#!/usr/bin/env python3
"""Run the paper's evaluation up to the predictions, as configs/run.yaml sets it.

    python scripts/run.py                                  # every step in configs/run.yaml
    python scripts/run.py --set datasets=[cardiobench] --set steps.segmentation=false
    python scripts/run.py --dry-run                        # every step on 5 studies per split of each cohort
    python scripts/run.py --print                          # print the commands only
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import datasets_config, embedding_files, load_cohort  # noqa: E402

ZEROSHOT = ("echoalign", "echoprime", "panecho", "echoclip")
PROMPT_ARMS = ("echoprime", "echoclip")
SEGMENTATION = ("echoalign", "echoprime", "panecho", "echoclip")


def load_config(path, overrides):
    cfg = yaml.safe_load(open(path))
    for item in overrides:
        key, _, value = item.partition("=")
        node, *parts = key.split(".")
        target = cfg
        for p in [node, *parts][:-1]:
            target = target.setdefault(p, {})
        target[([node, *parts])[-1]] = yaml.safe_load(value)
    return cfg


class Runner:
    def __init__(self, print_only):
        self.print_only, self.failed = print_only, []

    def run(self, name, log, cmd, env=None):
        cmd = [str(c) for c in cmd]
        if self.print_only:
            print(f"# {name} > {log}\n{shlex.join(cmd)}", flush=True)
            return True
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).returncode
        return self._done(name, rc)

    def run_parallel(self, jobs, env=None):
        if self.print_only:
            return {name for name, log, cmd in jobs if self.run(name, log, cmd)}
        procs = []
        for name, log, cmd in jobs:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            f = open(log, "w")
            procs.append((name, f, subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                                                    env={**os.environ, **(env or {})})))
        ok = set()
        for name, f, p in procs:
            rc = p.wait()
            f.close()
            if self._done(name, rc):
                ok.add(name)
        return ok

    def _done(self, name, rc):
        print(f"{name} exit={rc}", flush=True)
        if rc:
            self.failed.append(name)
        return rc == 0


def py(script, *args):
    return [sys.executable, "-u", ROOT / "scripts" / script, *args]


def encoders(cfg):
    return (cfg.get("echoprime_encoders") or {}).items()


def embed(r, cfg, c, emb_dir, logs):
    n, only = int(cfg["embed_shards"]), cfg["embed_shard"]
    for name, m, extra in [(m, m, []) for m in cfg["models"]] + [
            (v, "echoprime", ["--echoprime-encoder", p]) for v, p in encoders(cfg)]:
        out = emb_dir / f"{name}.npz"
        cmd = py("embed.py", "--method", m, "--manifest", c["manifest"], "--out", out, "--weights", cfg["weights"],
                 "--echoprime-repo", cfg["repos"]["echoprime"],
                 "--panecho-source", cfg["repos"]["panecho"], "--batch-size", cfg["batch_size"],
                 "--device", cfg["device"], *extra, *cfg["embed_args"])
        if m == "echoprime":
            cmd += ["--echoprime-stills", c.get("echoprime_stills", "skip")]
        if n == 1:
            r.run(f"embed {name}", logs / f"embed_{name}.log", cmd)
            continue
        for i in range(n):
            if only is not None and i != int(only):
                continue
            r.run(f"embed {name} shard {i}", logs / f"embed_{name}_shard{i}.log",
                  cmd + ["--shard", i, "--num-shards", n])


def zeroshot(r, cfg, name, c, emb_dir, out, logs, tasks):
    zs = out / "zeroshot"
    common = ["--cohort", name, "--device", cfg["device"], "--weights", cfg["weights"],
              "--echoprime-repo", cfg["repos"]["echoprime"], "--echoclip-repo", cfg["repos"]["echoclip"],
              "--panecho-source", cfg["repos"]["panecho"], *tasks]
    inputs = []
    for m in [m for m in cfg["models"] if m in ZEROSHOT]:
        emb = emb_dir / f"{m}.npz"
        own = zs / f"{m}_own.csv"
        if not r.run(f"zs {m}", logs / f"zs_{m}.log", py("zeroshot.py", *common, "--method", m, "--embeddings", emb,
                                                          "--out", own)):
            continue
        files = [own]
        if m in PROMPT_ARMS:
            extra = ["--echoprime-pooling", "sections"] if m == "echoprime" else []
            if (r.run(f"zs {m}_prompts", logs / f"zs_{m}_prompts.log",
                      py("zeroshot.py", *common, "--method", f"{m}_prompts", "--embeddings", emb, *extra,
                         "--out", zs / f"{m}_prompts_all.csv"))
                    and r.run(f"zs {m} prompt subset", logs / f"zs_{m}_prompt_subset.log",
                              py("compare_inputs.py", "subset", "--scores", zs / f"{m}_prompts_all.csv",
                                 "--exclude-tasks-of", own, "--out", zs / f"{m}_prompts.csv"))):
                files.append(zs / f"{m}_prompts.csv")
        inputs += ["--method", f"{m}=" + ",".join(map(str, files))]
    if inputs:
        ref = ["--reference", "echoalign"] if any(i.startswith("echoalign=") for i in inputs) else []
        r.run("zs prepare", logs / "zs_prepare.log",
              py("compare_inputs.py", "prepare", "--labels", c["zeroshot_labels"], *ref, *inputs, "--out", zs))


def sections_match(sections, emb):
    if not sections.exists():
        return False
    try:
        files = embedding_files(emb)
    except (FileNotFoundError, ValueError):
        return False
    studies = set()
    for f in files:
        with np.load(f) as z:
            studies.update(z["study_id"].tolist())
    with np.load(sections) as z:
        return set(z["study_id"].tolist()) == studies


def probes(r, cfg, name, c, emb_dir, out, logs, tasks):
    threads = str(cfg["threads"])
    env = {k: threads for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    common = ["--labels", c["probe_labels"], "--threads", threads, "--seed-key", c["seed_key"], *tasks]
    jobs = [(f"probe {m}", logs / f"probe_{m}.log",
             py("probe.py", "--method", m, "--embeddings", emb_dir / f"{m}.npz", "--weights", cfg["weights"],
                *common, "--out", out / "probe"))
            for m in cfg["models"] if m != "echoprime"]
    r.run_parallel(jobs, env)
    variants = [("echoprime", "echoprime_sections")] * ("echoprime" in cfg["models"])
    variants += [(v, v) for v, _ in encoders(cfg)]
    for m, prefix in variants:
        sections = emb_dir / f"{m}_sections.npz"
        if sections_match(sections, emb_dir / f"{m}.npz") or r.run(f"{m} sections", logs / f"{m}_sections.log",
                                      py("echoprime_section_features.py", "--embeddings", emb_dir / f"{m}.npz",
                                         "--echoprime-repo", cfg["repos"]["echoprime"], "--out", sections)):
            r.run(f"probe {m}", logs / f"probe_{m}.log",
                  py("probe_echoprime_sections.py", "--features", sections, "--cohort", name, "--missing", "drop",
                     "--name", prefix, "--out", out / "echoprime_sections",
                     "--probe-out", out / "probe", "--probe-method", m, *common), env)


def subset_cohort(c, n, folder):
    folder.mkdir(parents=True, exist_ok=True)
    probe = pd.read_csv(c["probe_labels"], dtype=str)
    manifest = pd.read_csv(c["manifest"], dtype=str)
    probe = probe[probe.study_id.isin(manifest.study_id)]
    keep = set()
    for _, g in probe.groupby("split"):
        keep |= set(sorted(g.study_id.unique())[:n])
    out = dict(c)
    for key, df in (("manifest", manifest), ("probe_labels", probe),
                    ("zeroshot_labels", pd.read_csv(c["zeroshot_labels"], dtype=str))):
        out[key] = str(folder / f"{key}.csv")
        df[df.study_id.isin(keep)].to_csv(out[key], index=False)
    return out


def segmentation(r, cfg, limit=None):
    models = [m for m in cfg["models"] if m in SEGMENTATION]
    if not models:
        return
    out = ROOT / cfg["seg_out"]
    seg = py("segment.py", "--weights", cfg["weights"], "--echoprime-repo", cfg["repos"]["echoprime"],
             "--disable-cudnn", "--workers", 15, "--out", out, "--config", datasets_config(),
             "--models", *models, *(["--limit", limit] if limit else []), *cfg["seg_args"])
    r.run("segmentation prep and features", out / "logs/prep_extract.log", seg + ["--steps", "prep", "extract"])
    r.run("segmentation camus and transfer", out / "logs/train_camus_transfer.log",
          seg + ["--steps", "train", "--tasks", "camus", "transfer"])
    r.run("segmentation echonet", out / "logs/train_echonet.log",
          seg + ["--steps", "train", "--tasks", "echonet_dynamic", "echonet_pediatric"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/run.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    ap.add_argument("--dry-run", type=int, nargs="?", const=5, default=None, metavar="N",
                    help="run every step on N studies per split of each cohort (default 5)")
    ap.add_argument("--print", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    if a.dry_run:
        cfg["emb_root"] = f"{cfg['emb_root']}/dry_run"
        cfg["out_root"] = f"{cfg['out_root']}/dry_run"
        cfg["seg_out"] = f"{cfg['out_root']}/segmentation"
    os.chdir(ROOT)
    r = Runner(a.print)
    tasks = ["--tasks", *cfg["tasks"]] if cfg["tasks"] else []
    code = subprocess.run(["git", "describe", "--always", "--dirty", "--abbrev=40"],
                          capture_output=True, text=True).stdout.strip()
    print(f"datasets={cfg['datasets']} models={cfg['models']} weights={cfg['weights']} code={code}", flush=True)

    for name in cfg["datasets"]:
        c = load_cohort(name)
        emb_dir, out = ROOT / cfg["emb_root"] / name, ROOT / cfg["out_root"] / name
        if a.dry_run:
            c = subset_cohort(c, a.dry_run, out / "inputs")
        logs = out / "logs"
        print(f"== {name}", flush=True)
        if cfg["steps"]["embed"]:
            embed(r, cfg, c, emb_dir, logs)
            if cfg["embed_shard"] is not None:
                print(f"{name}: shard {cfg['embed_shard']} embedded; later steps skipped")
                continue
        if cfg["steps"]["zeroshot"]:
            zeroshot(r, cfg, name, c, emb_dir, out, logs, tasks)
        if cfg["steps"]["probes"]:
            probes(r, cfg, name, c, emb_dir, out, logs, tasks)
    if cfg["steps"]["segmentation"] and cfg["embed_shard"] is None:
        segmentation(r, cfg, a.dry_run)

    for name in r.failed:
        print(f"failed: {name}")
    sys.exit(1 if r.failed else 0)


if __name__ == "__main__":
    main()
