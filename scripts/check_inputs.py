#!/usr/bin/env python3
"""Show what every model receives for each clip, in true pixel values; for EchoPrime and EchoCLIP, next to the upstream code's input.

    python scripts/check_inputs.py --manifest clips.csv --out-dir results/input_check
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from echoalign.loader import file_kind, load_echo  # noqa: E402

IMAGENET = (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
CLIP = (np.array([0.48145466, 0.4578275, 0.40821073]), np.array([0.26862954, 0.26130258, 0.27577711]))
ALL = ("echoalign", "echoprime", "panecho", "echoclip", "echofm", "echojepa")


def _frames(x, layout):
    x = torch.as_tensor(x).float().cpu().numpy()
    return x.transpose(0, 2, 3, 1) if layout == "TCHW" else x.transpose(1, 2, 3, 0)


def received(name, x, path):
    if name == "echoalign":
        f, note = (_frames(x, "TCHW") * 0.5 + 0.5) * 255, "RGB, mean=std=0.5"
    elif name == "echoprime":
        mean = np.array([29.110628, 28.076836, 29.096405]); std = np.array([47.989223, 46.456997, 47.20083])
        f = _frames(x, "CTHW") * std + mean
        bgr = file_kind(Path(path)) == "dicom"
        f, note = (f[..., ::-1] if bgr else f), ("BGR (DICOM)" if bgr else "RGB") + ", EchoPrime mean/std"
    elif name == "panecho":
        f = (_frames(x, "CTHW") * IMAGENET[1] + IMAGENET[0]) * 255
        f, note = f[..., ::-1], "BGR, ImageNet"
    elif name == "echoclip":
        t, n = x
        f = (_frames(t[:n], "TCHW") * CLIP[1] + CLIP[0]) * 255
        f, note = f[..., ::-1], f"BGR, CLIP mean/std, {n} frames"
    elif name == "echofm":
        f, note = _frames(x, "CTHW") * 255, "RGB, [0, 1]"
    elif name == "echojepa":
        f, note = (_frames(x, "CTHW") * IMAGENET[1] + IMAGENET[0]) * 255, "RGB, ImageNet"
    return f, note


def build(name, a):
    from eval.embed import build as build_embedder
    # PanEcho and EchoJEPA both ship a top-level `src` package; drop the cached one so each imports its own.
    for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
        del sys.modules[k]
    kw = dict(device="cpu")
    if name == "echoprime":
        kw.update(repo_dir=a.echoprime_repo, stills=a.echoprime_stills)
    elif name == "panecho":
        kw.update(source=a.panecho_source, local=True)
    elif name == "echoalign":
        return None
    return build_embedder(name, **kw)


def upstream_echoprime(m, path, work_dir):
    import pydicom
    read = pydicom.dcmread

    def dcmread(*args, **kw):
        ds = read(*args, **kw)
        if int(pydicom.__version__.split(".")[0]) >= 3:
            ds.pixel_array_options(as_rgb=False)
        return ds
    with tempfile.TemporaryDirectory(dir=work_dir) as d:
        Path(d, Path(path).name).symlink_to(Path(path).resolve())
        pydicom.dcmread = dcmread
        try:
            x = type(m.ep).process_dicoms(m.ep, d)[0]
        except Exception as e:
            return None, f"upstream: {type(e).__name__}"
        finally:
            pydicom.dcmread = read
    return received("echoprime", x, path)[0], "upstream process_dicoms"


def upstream_echoclip(m, path, repo_dir):
    from torchvision import transforms as T
    from eval.zeroshot.echoclip import load_echoclip_repo
    utils, _ = load_echoclip_repo(repo_dir)
    v = utils.read_avi(path, (224, 224))
    if not len(v):
        return None, "upstream: no frames"
    x = torch.stack([m.preprocess_val(T.ToPILImage()(fr)) for fr in v])[0:min(40, len(v)):2]
    return received("echoclip", (x, len(x)), path)[0], "upstream read_avi + preprocess_val"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="CSV with clip_id and path (a label column is optional)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=list(ALL), choices=ALL)
    ap.add_argument("--echoprime-repo", default=str(ROOT / "models" / "EchoPrime"))
    ap.add_argument("--echoprime-stills", choices=["skip", "repeat"], default="skip")
    ap.add_argument("--echoclip-repo", default=str(ROOT / "models" / "echo_CLIP"))
    ap.add_argument("--panecho-source", default=str(ROOT / "models" / "PanEcho"))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    models = {}
    for n in a.models:
        try:
            models[n] = build(n, a)
        except Exception as e:
            print(f"{n}: not loaded ({type(e).__name__}: {e})")
    man = pd.read_csv(a.manifest, dtype=str)
    rows_in, rows_up = [], []
    for row in man.itertuples():
        label = getattr(row, "label", row.clip_id)
        panels = []
        for n in a.models:
            if n not in models:
                panels += [(f"{n}", None, "not loaded")] * 2
                continue
            try:
                x = load_echo(row.path) if n == "echoalign" else models[n].preprocess(row.path)
                if x is None:
                    panels += [(f"{n}", None, "rejected (still image)")] * 2
                    continue
                f, note = received(n, x, row.path)
                rows_in.append(dict(clip=row.clip_id, model=n, frames=len(f), height=f.shape[1], width=f.shape[2],
                                    min=float(f.min()), max=float(f.max()), note=note))
                panels += [(f"{n}  frame 0", f[0], note), (f"{n}  frame {len(f) // 2}", f[len(f) // 2], "")]
                if n == "echoprime" and file_kind(Path(row.path)) == "dicom":
                    g, gnote = upstream_echoprime(models[n], row.path, out)
                elif n == "echoclip" and file_kind(Path(row.path)) == "video":
                    g, gnote = upstream_echoclip(models[n], row.path, a.echoclip_repo)
                else:
                    continue
                if g is None:
                    panels += [(f"{n} upstream", None, gnote)]
                    rows_up.append(dict(clip=row.clip_id, model=n, comparison=gnote))
                    continue
                k = min(len(f), len(g))
                d = np.abs(f[:k] - g[:k])
                rows_up.append(dict(clip=row.clip_id, model=n, comparison=gnote, frames_ours=len(f), frames_upstream=len(g),
                                    max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean())))
                panels += [(f"{n} upstream  frame 0", g[0], gnote),
                           (f"|diff| frame 0 (max {d[0].max():.2f})", np.clip(d[0] * 50, 0, 255), "x50")]
            except Exception as e:
                panels += [(f"{n}", None, f"{type(e).__name__}: {str(e)[:60]}")] * 2
        cols = 4
        nrow = (len(panels) + cols - 1) // cols
        fig, axes = plt.subplots(nrow, cols, figsize=(3.1 * cols, 3.3 * nrow), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (title, img, note) in zip(axes.ravel(), panels):
            if img is None:
                ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=8, wrap=True)
            else:
                ax.imshow(np.clip(img, 0, 255).astype(np.uint8))
            ax.set_title(title + (f"\n{note}" if note else ""), fontsize=7.5)
        fig.suptitle(f"{label}: {Path(row.path).name}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{row.clip_id}.png", dpi=90)
        plt.close(fig)
        print(f"{row.clip_id}: {len(panels)} panels")
    pd.DataFrame(rows_in).to_csv(out / "inputs.csv", index=False)
    pd.DataFrame(rows_up).to_csv(out / "upstream.csv", index=False)
    print(pd.DataFrame(rows_up).to_string(index=False))


if __name__ == "__main__":
    main()
