#!/usr/bin/env python3
"""Embed every clip of a manifest with one model and write a .npz file.

    python scripts/embed.py --method echoalign --manifest data/manifests/mimic_iv_echo.csv \
        --out embeddings/mimic_iv_echo/echoalign.npz
    python scripts/embed.py --method echoprime --echoprime-repo models/EchoPrime \
        --manifest data/manifests/mimic_iv_echo.csv --out embeddings/mimic_iv_echo/echoprime.npz
    python scripts/embed.py --method echoprime --echoprime-repo models/EchoPrime \
        --echoprime-encoder weights/my_echoprime_encoder.pt \
        --manifest data/manifests/mimic_iv_echo.csv --out embeddings/mimic_iv_echo/my_echoprime.npz
    python scripts/embed.py --method panecho --cohort mimic_iv_echo --shard 3/8
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.data import SHARD_NAME, embedding_files, load_cohort, shard_path  # noqa: E402
from eval.embed import METHODS  # noqa: E402

CHECK_STUDIES = 2
ATOL = 1e-4  # GPU kernels are not bit-exact


def parse_shard(text):
    i, _, n = text.partition("/")
    try:
        return int(i), (int(n) if n else None)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard must be i or i/n, not {text!r}")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=METHODS, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest")
    src.add_argument("--cohort", help="cohort in configs/cohorts.yaml: its manifest, output under --emb-root")
    ap.add_argument("--out", default=None, help="output .npz (required with --manifest)")
    ap.add_argument("--emb-root", default=str(ROOT / "embeddings"), help="--cohort: default output <emb-root>/<cohort>/<method>.npz")
    ap.add_argument("--weights", default=str(ROOT / "weights/EchoAlign.pt"))
    ap.add_argument("--echoprime-repo", default=str(ROOT / "models/EchoPrime"), help="clone of EchoPrime")
    ap.add_argument("--echoprime-encoder", default=None,
                    help="EchoPrime video encoder state_dict to use instead of the released one (same architecture)")
    ap.add_argument("--echoprime-stills", choices=["skip", "repeat"], default="skip",
                    help="EchoPrime: skip still images, as EchoPrime's own pipeline does, or repeat each to a "
                         "32-frame clip (see eval/embed/echoprime.py)")
    ap.add_argument("--panecho-source", default=str(ROOT / "models/PanEcho"), help="PanEcho source repository or local clone")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--shard", type=parse_shard, default=None,
                    help="0-based shard to embed: i (with --num-shards) or i/n")
    ap.add_argument("--num-shards", type=int, default=None, help="split the manifest's studies into this many shards")
    a = ap.parse_args()
    if a.manifest and not a.out:
        ap.error("--manifest needs --out")
    if a.cohort:
        a.manifest = load_cohort(a.cohort)["manifest"]
        a.out = a.out or str(Path(a.emb_root) / a.cohort / f"{a.method}.npz")
    shard, n = a.shard if a.shard else (None, None)
    if n and a.num_shards and n != a.num_shards:
        ap.error(f"--shard {shard}/{n} disagrees with --num-shards {a.num_shards}")
    if shard is not None and not (n or a.num_shards):
        ap.error("--shard needs a shard count: i/n or --num-shards")
    a.shard, a.num_shards = shard, n or a.num_shards or 1
    return a


def missing_shards(out):
    files = sorted(Path(out).parent.glob(f"{Path(out).stem}.shard-*-of-*.npz"))
    totals = {int(SHARD_NAME.search(f.name).group(2)) for f in files}
    if len(totals) != 1:
        return [f.name for f in files]
    n = totals.pop()
    return [shard_path(out, i, n).name for i in range(n) if not shard_path(out, i, n).exists()]


def mismatch(a, files):
    import numpy as np
    import pandas as pd
    from eval.embed.run import create_embeddings

    stored = [np.load(f) for f in files]
    df = pd.read_csv(a.manifest, dtype=str)
    studies = sorted(set(df.study_id) & set(np.concatenate([z["study_id"] for z in stored])))[:CHECK_STUDIES]
    if not studies:
        return "no study of the manifest is in the file"
    fresh = create_embeddings(manifest=df[df.study_id.isin(studies)], method=a.method, batch_size=a.batch_size,
                              num_workers=a.workers, device=a.device, save_clips=False, **model_kwargs(a))
    for name, v in fresh.items():
        if v.dtype.kind not in "fc":
            continue
        key = "clip_id" if "clip_id" in fresh and len(fresh["clip_id"]) == len(v) else "study_id"
        old = {}
        for z in stored:
            if name not in z.files:
                return f"array {name} is not in {files[0].name}"
            hit = np.isin(z[key], fresh[key])
            old.update(zip(z[key][hit], z[name][hit]))
        if set(old) != set(fresh[key]):
            return f"{name}: {len(set(fresh[key]) - set(old))} of {len(v)} rows are not in the file"
        diff = max(float(np.abs(np.asarray(old[k], np.float64) - x).max()) for k, x in zip(fresh[key], v))
        if diff > ATOL:
            return f"{name} differs by up to {diff:.3g} (allowed {ATOL})"
    return None


def reuse(a):
    if a.shard is not None:
        target = shard_path(a.out, a.shard, a.num_shards) if a.num_shards > 1 else Path(a.out)
        if not target.exists():
            return False
        files = [target]
    else:
        try:
            files = embedding_files(a.out)
        except FileNotFoundError:
            return False
        except ValueError:
            raise SystemExit(f"incomplete shard set for {a.out}; missing: {missing_shards(a.out)}")
    why = mismatch(a, files)
    if why:
        raise SystemExit(f"{files[0]} does not match {a.method} with these weights ({why}); move it away to embed again")
    print("reuse " + " ".join(str(f) for f in files))
    return True


def model_kwargs(a):
    kwargs = {}
    if a.method == "echoalign":
        kwargs["weights_path"] = a.weights
    elif a.method == "echoprime":
        kwargs["repo_dir"] = a.echoprime_repo
        if a.echoprime_encoder:
            kwargs["encoder_path"] = a.echoprime_encoder
        kwargs["stills"] = a.echoprime_stills
    elif a.method == "panecho":
        kwargs["source"] = a.panecho_source
        kwargs["local"] = True
    return kwargs


def main():
    a = parse_args()
    if reuse(a):
        return

    from eval.embed.run import create_embeddings

    create_embeddings(
        manifest=a.manifest,
        method=a.method,
        out_path=a.out,
        batch_size=a.batch_size,
        num_workers=a.workers,
        device=a.device,
        shard=a.shard,
        num_shards=a.num_shards,
        **model_kwargs(a),
    )
    print(f"Embedding completed for {a.method} -> {a.out}")


if __name__ == "__main__":
    main()
