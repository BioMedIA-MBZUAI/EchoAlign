import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from echoalign.device import check_miopen
from eval.analysis.data import shard_path
from . import METHODS, build


class ClipsDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: pd.DataFrame, preprocess_fn):
        self.manifest = manifest
        self.preprocess_fn = preprocess_fn

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        path = self.manifest.path.iloc[idx]
        try:
            x = self.preprocess_fn(path)
            error = None if x is not None else REJECTED
        except Exception as e:
            x, error = None, f"{type(e).__name__}: {e}"
        return idx, x, error


REJECTED = "rejected by the model's pipeline"


def _collate_clips(items):
    keep = [(i, x) for i, x, e in items if x is not None]
    errors = [(i, e) for i, x, e in items if x is None]
    if not keep:
        return [], None, errors
    idx, xs = zip(*keep)
    if isinstance(xs[0], tuple):
        return list(idx), (torch.stack([x[0] for x in xs]), [x[1] for x in xs]), errors
    return list(idx), torch.stack(xs), errors


def create_embeddings(
    manifest: Union[str, Path, pd.DataFrame],
    method: str = "echoalign",
    out_path: Optional[Union[str, Path]] = None,
    weights_path: str = "weights/EchoAlign.pt",
    batch_size: int = 16,
    num_workers: int = 4,
    device: Optional[str] = None,
    save_clips: bool = True,
    shard: Optional[int] = None,
    num_shards: int = 1,
    log_every: float = 60.0,
    **kwargs,
) -> Dict[str, np.ndarray]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {METHODS}")

    if isinstance(manifest, (str, Path)):
        df = pd.read_csv(manifest, dtype=str)
    else:
        df = manifest.copy()

    missing = {"clip_id", "study_id", "path"} - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    if df.clip_id.duplicated().any():
        raise ValueError("Manifest contains duplicate clip_id entries.")
    if num_shards > 1:
        if shard is None or not 0 <= shard < num_shards:
            raise ValueError(f"shard must be in [0, {num_shards}) when num_shards > 1")
        studies = np.sort(df.study_id.unique())
        mine = set(studies[shard::num_shards])
        df = df[df.study_id.isin(mine)].reset_index(drop=True)
        if out_path:
            base = out_path
            out_path = shard_path(base, shard, num_shards)
            clips_base = Path(base).with_name(Path(base).stem + "_clips.npz")
    if df.empty:
        raise ValueError("No clips to embed.")

    model_kwargs = kwargs.copy()
    if method == "echoalign":
        model_kwargs.setdefault("weights", weights_path)
    elif method == "echoprime":
        model_kwargs.setdefault("repo_dir", "models/EchoPrime")
    elif method == "panecho":
        model_kwargs.setdefault("source", "models/PanEcho")
        model_kwargs.setdefault("local", True)

    model_kwargs.setdefault("device", device)
    if str(device or "cuda").startswith("cuda"):
        check_miopen()

    embedder = build(method, **model_kwargs)
    dataset = ClipsDataset(df, embedder.preprocess)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate_clips,
    )

    rows, parts, skipped, reasons = [], {}, [], []
    total, start = len(df), time.time()
    last = start
    for n, (idx, batch, errors) in enumerate(loader):
        for i, e in errors:
            skipped.append(df.clip_id.iloc[i])
            reasons.append(e)
        if idx:
            for k, v in embedder.encode(batch).items():
                parts.setdefault(k, []).append(v)
            rows += idx
        now = time.time()
        if log_every and now - last >= log_every:
            done = len(rows) + len(skipped)
            rate = done / (now - start)
            left = (total - done) / rate if rate else float("nan")
            print(f"{method}: {done}/{total} clips ({len(skipped)} skipped), {rate:.1f} clips/s, "
                  f"{left / 60:.0f} min left", flush=True)
            last = now

    failed = [r for r in reasons if r != REJECTED]
    print(f"{method}: {len(rows)} embedded, {len(reasons) - len(failed)} rejected by the model, "
          f"{len(failed)} failed to load", flush=True)
    for r in failed[:5]:
        print(f"  {r}", flush=True)
    if not rows:
        raise RuntimeError(f"{method}: no clip could be embedded; check the manifest paths")

    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["clip_id"] = df.clip_id.to_numpy()[rows].astype(str)
    out["study_id"] = df.study_id.to_numpy()[rows].astype(str)
    out["skipped"] = np.asarray(skipped, dtype=str)
    out["skip_reason"] = np.asarray(reasons, dtype=str)

    if hasattr(embedder, "study_level"):
        if out_path and save_clips:
            clips_path = (shard_path(clips_base, shard, num_shards) if num_shards > 1
                          else Path(out_path).with_name(Path(out_path).stem + "_clips.npz"))
            Path(clips_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(clips_path, **out)
        out = embedder.study_level(out, out["study_id"])
        out["skipped"] = np.asarray(skipped, dtype=str)
        out["skip_reason"] = np.asarray(reasons, dtype=str)

    if out_path:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_file, **out)
        if skipped:
            pd.DataFrame({"clip_id": skipped, "reason": reasons}).to_csv(
                out_file.with_name(out_file.stem + "_skipped.csv"), index=False)
            print(f"skipped clips and reasons: {out_file.with_name(out_file.stem + '_skipped.csv')}", flush=True)

    return out


__all__ = ["create_embeddings", "ClipsDataset"]
