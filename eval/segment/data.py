import json
from pathlib import Path

import numpy as np
import pandas as pd

from echoalign.loader import count_frames, file_kind, probe_video, read_frames, video_fps

CLASSES = {
    "camus": ["background", "LV_cavity", "LV_myocardium", "LA"],
    "echonet_dynamic": ["background", "LV_cavity"],
    "echonet_pediatric": ["background", "LV_cavity"],
}
DATASETS = tuple(CLASSES)
K_FRAMES = 16
COLUMNS = ["clip", "patient", "split", "view", "video_path", "n_frames", "ed_idx", "es_idx", "ed_ok", "es_ok", "H", "W"]


def dataset_roots(config):
    import yaml
    with open(config) as f:
        cfg = yaml.safe_load(f) or {}
    return {k: (Path(cfg[k]).expanduser() if cfg.get(k) else None) for k in DATASETS}


def check_mask(m):
    import cv2
    area = int(m.sum())
    if area == 0:
        return False
    _, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return bool(area >= 0.01 * m.size and stats[1:, cv2.CC_STAT_AREA].max() / area >= 0.99)


def _camus_gt(path):
    import nibabel as nib
    g = np.rint(np.asarray(nib.load(str(path)).dataobj)).astype(np.uint8)
    return np.ascontiguousarray(np.squeeze(np.swapaxes(g, 0, 1)))


def _camus_one(args):
    import nibabel as nib
    root, patient, view, split = args
    base = Path(root) / "database_nifti" / patient / f"{patient}_{view}"
    video = f"{base}_half_sequence.nii.gz"
    seq = np.asarray(nib.load(video).dataobj)
    idx = {}
    for ph in ("ED", "ES"):
        im = np.squeeze(np.asarray(nib.load(f"{base}_{ph}.nii.gz").dataobj))
        hit = [t for t in range(seq.shape[-1]) if np.array_equal(im, seq[..., t])]
        if not hit:
            raise RuntimeError(f"{base}_{ph}.nii.gz matches no half-sequence frame")
        idx[ph] = hit[0] if ph == "ED" else hit[-1]
    ed, es = _camus_gt(f"{base}_ED_gt.nii.gz"), _camus_gt(f"{base}_ES_gt.nii.gz")
    row = dict(clip=f"{patient}_{view}", patient=patient, split=split, view=view, video_path=video,
               n_frames=int(seq.shape[-1]), ed_idx=idx["ED"], es_idx=idx["ES"], ed_ok=True, es_ok=True,
               H=int(ed.shape[0]), W=int(ed.shape[1]))
    return row, ed, es


def camus_clip(root, patient, view, split="TRAIN"):
    return _camus_one((str(root), patient, view, split))


def camus_jobs(root):
    root = Path(root)
    jobs = []
    for split, name in (("TRAIN", "training"), ("VAL", "validation"), ("TEST", "testing")):
        for p in (root / "database_split" / f"subgroup_{name}.txt").read_text().split():
            jobs += [(str(root), p, v, split) for v in ("2CH", "4CH")]
    return jobs


def _echonet_mask(tr, h, w):
    import cv2
    x = np.concatenate((tr[1:, 0], np.flip(tr[1:, 2])))
    y = np.concatenate((tr[1:, 1], np.flip(tr[1:, 3])))
    pts = np.stack([np.rint(x), np.rint(y)], 1).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m


def _two_frames(masks, n, shift=0):
    ed_f, es_f = sorted(masks, key=lambda f: (-int(masks[f].sum()), f))
    out = {}
    for ph, f in (("ed", ed_f), ("es", es_f)):
        i = int(f) - shift
        out[ph] = (i, masks[f], bool(check_mask(masks[f]) and 0 <= i < n))
    return out


def _echonet_one(args):
    r, tr = args
    h, w, n = int(r["FrameHeight"]), int(r["FrameWidth"]), int(r["NumberOfFrames"])
    frames = list(dict.fromkeys(tr.Frame.tolist()))
    if len(frames) != 2:
        return None
    mk = {f: _echonet_mask(tr[tr.Frame == f][["X1", "Y1", "X2", "Y2"]].to_numpy(float), h, w) for f in frames}
    p = _two_frames(mk, n)
    clip = r["FileName"]
    row = dict(clip=clip, patient=clip, split=str(r["Split"]).upper(), view="A4C", video_path=r["video_path"],
               n_frames=n, ed_idx=p["ed"][0], es_idx=p["es"][0], ed_ok=p["ed"][2], es_ok=p["es"][2], H=h, W=w)
    return row, p["ed"][1], p["es"][1]


def echonet_jobs(root):
    root = Path(root)
    fl = pd.read_csv(root / "FileList.csv")
    fl["FileName"] = fl.FileName.astype(str).str.replace(r"\.avi$", "", regex=True)
    fl["video_path"] = [str(root / "Videos" / f"{f}.avi") for f in fl.FileName]
    vt = pd.read_csv(root / "VolumeTracings.csv")
    vt["FileName"] = vt.FileName.astype(str).str.replace(r"\.avi$", "", regex=True)
    trace = {k: g.reset_index(drop=True) for k, g in vt.groupby("FileName", sort=False)}
    return [(r, trace[r["FileName"]]) for r in fl.to_dict("records") if r["FileName"] in trace]


def _pediatric_one(args):
    import cv2
    r, tr, base = args
    w, h, n = probe_video(Path(r["video_path"]))
    if n <= 0:                                                            # no nb_frames in the container
        n = count_frames(Path(r["video_path"]), w, h)
    frames = list(dict.fromkeys(tr.Frame.tolist()))
    if len(frames) != 2:
        return None
    mk = {}
    for f in frames:
        pts = np.rint(tr[tr.Frame == f][["X", "Y"]].to_numpy(float)).astype(np.int32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [pts], 1)
        mk[f] = m
    p = _two_frames(mk, n, shift=base)
    split = _pediatric_split(int(r["Split"]))
    clip = r["FileName"]
    row = dict(clip=clip, patient="-".join(clip.split("-")[:2]), split=split, view="A4C", video_path=r["video_path"],
               n_frames=int(n), ed_idx=p["ed"][0], es_idx=p["es"][0], ed_ok=p["ed"][2], es_ok=p["es"][2], H=int(h), W=int(w))
    return row, p["ed"][1], p["es"][1]


def _pediatric_split(fold):
    return "TRAIN" if fold <= 7 else ("VAL" if fold == 8 else "TEST")


def pediatric_jobs(root, frame_base=1):
    a4c = Path(root) / "A4C"
    fl = pd.read_csv(a4c / "FileList.csv")
    fl["video_path"] = [str(a4c / "Videos" / f) for f in fl.FileName]
    vt = pd.read_csv(a4c / "VolumeTracings.csv")
    num = pd.to_numeric(vt.Frame, errors="coerce")
    vt = vt[num.notna() & vt.X.notna() & vt.Y.notna()].copy()
    vt["Frame"] = pd.to_numeric(vt.Frame).astype(int)
    trace = {k: g.reset_index(drop=True) for k, g in vt.groupby("FileName", sort=False)}
    return [(r, trace[r["FileName"]], int(frame_base)) for r in fl.to_dict("records") if r["FileName"] in trace]


def prepare(dataset, root, out_dir, workers=8, pediatric_frame_base=1, limit=None):
    from multiprocessing import Pool
    out_dir = Path(out_dir)
    mdir = out_dir / "masks" / dataset
    mdir.mkdir(parents=True, exist_ok=True)
    if dataset == "camus":
        fn, jobs = _camus_one, camus_jobs(root)
    elif dataset == "echonet_dynamic":
        fn, jobs = _echonet_one, echonet_jobs(root)
    elif dataset == "echonet_pediatric":
        fn, jobs = _pediatric_one, pediatric_jobs(root, pediatric_frame_base)
    else:
        raise ValueError(dataset)
    if limit:
        jobs = _limit_jobs(dataset, jobs, limit)
    with Pool(workers) as pool:
        res = [x for x in pool.imap(fn, jobs, chunksize=8) if x is not None]
    for row, ed, es in res:
        np.savez_compressed(mdir / f"{row['clip']}.npz", ed=ed.astype(np.uint8), es=es.astype(np.uint8))
    m = pd.DataFrame([r for r, _, _ in res], columns=COLUMNS).sort_values("clip").reset_index(drop=True)
    m.to_csv(out_dir / f"{dataset}_manifest.csv", index=False)
    t = image_table(m)
    summary = dict(dataset=dataset, clips=len(m), images=t.split.value_counts().to_dict(),
                   patients=t.groupby("split").patient.nunique().to_dict(), limit=limit,
                   pediatric_frame_base=pediatric_frame_base if dataset == "echonet_pediatric" else None)
    (out_dir / f"{dataset}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return m


def _limit_jobs(dataset, jobs, limit):
    def key(j):
        if dataset == "camus":
            return j[3], j[1]
        r = j[0]
        if dataset == "echonet_dynamic":
            return str(r["Split"]).upper(), r["FileName"]
        return _pediatric_split(int(r["Split"])), "-".join(r["FileName"].split("-")[:2])
    keep = {}
    for j in jobs:
        s, p = key(j)
        keep.setdefault(s, set()).add(p)
    keep = {s: set(sorted(v)[:limit]) for s, v in keep.items()}
    return [j for j in jobs if key(j)[1] in keep[key(j)[0]]]


def load_manifest(prep_dir, dataset):
    return pd.read_csv(Path(prep_dir) / f"{dataset}_manifest.csv", dtype={"patient": str, "clip": str})


def image_table(manifest):
    rows = []
    for r in manifest.to_dict("records"):
        for ph in ("ed", "es"):
            if not bool(r[f"{ph}_ok"]):
                continue
            rows.append(dict(image=f"{r['clip']}_{ph}", clip=r["clip"], patient=str(r["patient"]), split=r["split"],
                             view=r["view"], phase=ph, video_path=r["video_path"], n=int(r["n_frames"]),
                             a=int(r[f"{ph}_idx"]), H=int(r["H"]), W=int(r["W"])))
    return pd.DataFrame(rows).sort_values("image").reset_index(drop=True)


def native_mask(prep_dir, dataset, clip, phase):
    with np.load(Path(prep_dir) / "masks" / dataset / f"{clip}.npz") as z:
        return z[phase]


def window_around(n, a, stride, k=K_FRAMES):
    n, a, s = int(n), int(a), int(stride)
    assert 0 <= a < n
    lo, hi = max(0, k - 1 - (n - 1 - a) // s), min(k - 1, a // s)
    pos = min(max(k // 2, lo), hi) if lo <= hi else min(k // 2, hi)
    idx = [min(a + s * (j - pos), n - 1) for j in range(k)]
    assert idx[pos] == a and min(idx) >= 0
    return idx, pos


def resampled_window(path, n, a, target_fps, k=K_FRAMES):
    n, a = int(n), int(a)
    fps = video_fps(Path(path)) if file_kind(Path(path)) == "video" else None
    if not fps or not target_fps:
        return window_around(n, a, 1, k)
    step = fps / target_fps
    before = int(np.floor(a / step + 1e-9))
    times = np.arange(-before, int(np.floor((n - 1 - a) / step + 1e-9)) + 1)
    grid = np.clip(np.round(a + times * step).astype(int), 0, n - 1).tolist()
    sub, pos = window_around(len(grid), before, 1, k)
    return [grid[i] for i in sub], pos


def read_at(path, indices):
    idx = [int(i) for i in indices]
    frames = read_frames(path, n_frames=max(idx) + 1, stride=1, crop_to_region=False, pad_to_square=False)
    return [frames[i] for i in idx]
