from pathlib import Path

import numpy as np
import torch

from eval.embed.crop import crop_and_scale

from .data import read_at, resampled_window, window_around

MODELS = ("echoalign", "echoprime", "panecho", "echoclip")
STRIDE = {"echoprime": 2}
TARGET_FPS = {"echoalign": 24}
GRID = {"echoalign": 14, "echoprime": 7, "panecho": 7, "echoclip": 7}
CHANNELS = {"echoalign": 768, "echoprime": 768, "panecho": 768, "echoclip": 1024}
IMAGE_MODELS = ("panecho", "echoclip")
BATCH = {"echoalign": 16, "echoprime": 4, "panecho": 32, "echoclip": 32}
NORM = {"echoalign": ([0.5] * 3, [0.5] * 3),
        "panecho": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "echoclip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])}
EP_MEAN, EP_STD = [29.110628, 28.076836, 29.096405], [47.989223, 46.456997, 47.20083]
AUG = dict(scale=(0.9, 1.1), rot_deg=(-10.0, 10.0), contrast=(0.9, 1.1), brightness=(-12.75, 12.75))
SEED = 20261018


def echoprime_crop_box(h, w, zoom=0.1):
    y0, y1, x0, x1 = 0, h, 0, w
    if w / h > 1.0:
        p = int(round((w - h) / 2))
        if p > 0:
            x0, x1 = p, w - p
    if w / h < 1.0:
        p = int(round((h - w) / 2))
        if p > 0:
            y0, y1 = p, h - p
    px, py = round(int((x1 - x0) * zoom)), round(int((y1 - y0) * zoom))
    if px > 0 and py > 0:
        y0, y1, x0, x1 = y0 + py, y1 - py, x0 + px, x1 - px
    return y0, y1, x0, x1


def labels_to_model(model, lab):
    import cv2
    from PIL import Image
    if model != "echoprime":
        return np.asarray(Image.fromarray(lab.astype(np.uint8), mode="L").resize((224, 224), Image.NEAREST))
    y0, y1, x0, x1 = echoprime_crop_box(*lab.shape)
    return cv2.resize(np.ascontiguousarray(lab[y0:y1, x0:x1].astype(np.uint8)), (224, 224), interpolation=cv2.INTER_NEAREST)


def preprocess(model, frames):
    if model in NORM:
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode
        mean, std = NORM[model]
        out = []
        for f in frames:
            if f.size != (224, 224):
                f = TF.resize(f, (224, 224), interpolation=InterpolationMode.BICUBIC)
            out.append(TF.to_tensor(f))
        return TF.normalize(torch.stack(out, 0), mean, std)
    a = np.stack([crop_and_scale(np.asarray(f)) for f in frames]).astype(np.float32)
    x = torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()
    return (x - torch.tensor(EP_MEAN).view(1, 3, 1, 1)) / torch.tensor(EP_STD).view(1, 3, 1, 1)


def aug_params(row, version, seed=SEED):
    rng = np.random.default_rng([seed, int(row), int(version)])
    return dict(scale=float(rng.uniform(*AUG["scale"])), rot=float(rng.uniform(*AUG["rot_deg"])),
                contrast=float(rng.uniform(*AUG["contrast"])), brightness=float(rng.uniform(*AUG["brightness"])))


def augment(frames, mask, p):
    import cv2
    from PIL import Image
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), p["rot"], p["scale"])
    out = []
    for f in frames:
        a = cv2.warpAffine(np.asarray(f), M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = np.clip(a.astype(np.float32) * p["contrast"] + p["brightness"], 0, 255).astype(np.uint8)
        out.append(Image.fromarray(a))
    m = cv2.warpAffine(mask.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, m


def model_input(model, row, mask, version=0, seed=SEED):
    if model in IMAGE_MODELS:
        idx, pos = [row["a"]], 0
    elif model in TARGET_FPS:
        idx, pos = resampled_window(row["video_path"], row["n"], row["a"], TARGET_FPS[model])
    else:
        idx, pos = window_around(row["n"], row["a"], STRIDE[model])
    frames = read_at(row["video_path"], idx)
    if frames[pos].size != (mask.shape[1], mask.shape[0]):
        raise RuntimeError(f"frame size {frames[pos].size} differs from mask {mask.shape} for {row['video_path']}")
    if version > 0:
        frames, mask = augment(frames, mask, aug_params(row["row"], version, seed))
    return preprocess(model, frames), pos, labels_to_model(model, mask).astype(np.uint8)


class FeatureMaps:
    def __init__(self, model, device=None, weights="weights/EchoAlign.pt", echoprime_repo="models/EchoPrime",
                 echoprime_weights=None, panecho_source="models/PanEcho", panecho_local=True,
                 echoclip_model_id="hf-hub:mkaichristensen/echo-clip"):
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        cuda = self.device.type == "cuda"
        self.autocast = cuda and model in ("echoalign", "panecho", "echoclip")
        if model == "echoalign":
            from echoalign.encoder import load_clip_encoder
            blob = torch.load(weights, map_location="cpu", weights_only=True)
            self.net = load_clip_encoder(blob["state_dict"], self.device)
        elif model == "echoprime":
            import torchvision
            m = torchvision.models.video.mvit_v2_s()
            m.head[-1] = torch.nn.Linear(m.head[-1].in_features, 512)
            w = echoprime_weights or Path(echoprime_repo) / "model_data/weights/echo_prime_encoder.pt"
            m.load_state_dict(torch.load(w, map_location="cpu"), strict=True)
            self.net = m.to(self.device).eval()
        elif model == "panecho":
            from eval.zeroshot.panecho import hub_load
            enc = hub_load(str(panecho_source), panecho_local, image_encoder_only=True)
            self.net = enc.model.to(self.device).eval()
        elif model == "echoclip":
            import open_clip
            net, _, _ = open_clip.create_model_and_transforms(echoclip_model_id, precision="bf16" if cuda else "fp32",
                                                              device=self.device)
            self.net = net.visual.trunk.eval()
        else:
            raise ValueError(model)

    def _mvit(self, x):
        m = self.net
        h = m.conv_proj(x.permute(0, 2, 1, 3, 4).contiguous()).flatten(2).transpose(1, 2)
        h = m.pos_encoding(h)
        thw = (m.pos_encoding.temporal_size,) + m.pos_encoding.spatial_size
        for blk in m.blocks:
            h, thw = blk(h, thw)
        h = m.norm(h)
        t, hh, ww = thw
        return h[:, 1:].reshape(h.shape[0], t, hh, ww, h.shape[-1])

    @torch.inference_mode()
    def maps(self, x, pos):
        x = x.to(self.device, non_blocking=True)
        B, T = x.shape[:2]
        pos = [int(p) for p in pos]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast):
            if self.model == "echoalign":
                _, _, tok = self.net(x, return_tokens=True)
                tok = tok.reshape(B, 196, T, tok.shape[-1])
                f = torch.stack([tok[b, :, pos[b]] for b in range(B)], 0)
                return f.float().transpose(1, 2).reshape(B, -1, 14, 14)
            if self.model == "echoprime":
                tok = self._mvit(x)
                f = torch.stack([tok[b, pos[b] // 2] for b in range(B)], 0)
                return f.float().permute(0, 3, 1, 2).contiguous()
            fr = torch.stack([x[b, pos[b]] for b in range(B)], 0)
            if self.model == "panecho":
                return self.net.features(fr).float()
            t = self.net
            h = t.stem(fr.to(next(t.parameters()).dtype))
            for s in t.stages:
                h = s(h)
            return t.norm_pre(h).float()


def extract(model, dataset, table, prep_dir, out_path, n_aug=4, seed=SEED, batch_size=None, workers=8,
            log=print, **encoder_kwargs):
    import h5py
    from torch.utils.data import DataLoader, Dataset

    from .data import native_mask

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    items = [(r, v) for r in range(len(table)) for v in range(n_aug + 1 if table.split.iloc[r] == "TRAIN" else 1)]
    n, C, g = len(items), CHANNELS[model], GRID[model]
    rows = np.array([i[0] for i in items], np.int32)
    vers = np.array([i[1] for i in items], np.int8)
    with h5py.File(out_path, "a") as f:
        if "done" in f and not (len(f["row"]) == n and np.array_equal(f["row"][:], rows) and np.array_equal(f["ver"][:], vers)):
            raise RuntimeError(f"{out_path} was written for different images or n_aug; delete it to rebuild")
        if "done" not in f:
            f["row"], f["ver"] = rows, vers
            f.create_dataset("f", (n, C, g, g), np.float16, chunks=(1, C, g, g))
            f.create_dataset("lab", (n, 224, 224), np.uint8, chunks=(1, 224, 224), compression="gzip", compression_opts=1)
            f.create_dataset("done", (n,), bool, fillvalue=False)
            f.attrs.update(model=model, dataset=dataset, n_aug=n_aug, seed=seed, grid=g, channels=C)
        todo = np.flatnonzero(~f["done"][:])
    if len(todo) == 0:
        log(f"cache complete: {out_path}")
        return out_path
    recs = table.to_dict("records")

    class Items(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            k = int(todo[i])
            r, v = items[k]
            row = dict(recs[r], row=r)
            mask = native_mask(prep_dir, dataset, row["clip"], row["phase"])
            x, pos, lab = model_input(model, row, mask, v, seed)
            return k, x.numpy(), pos, lab                              # NumPy: no tensor sharing between processes

    def collate(b):
        return (np.array([i[0] for i in b]), np.stack([i[1] for i in b]), [i[2] for i in b],
                np.stack([i[3] for i in b]))

    enc = FeatureMaps(model, **encoder_kwargs)
    dl = DataLoader(Items(), batch_size=batch_size or BATCH[model], num_workers=workers, collate_fn=collate,
                    prefetch_factor=2 if workers else None)
    log(f"extract {model} {dataset}: {len(todo)} of {n} items to do -> {out_path}")
    with h5py.File(out_path, "a") as f:
        for b, (k, x, pos, lab) in enumerate(dl):
            v = enc.maps(torch.from_numpy(x), pos).cpu().numpy()
            if v.shape[1:] != (C, g, g) or not np.isfinite(v).all():
                raise RuntimeError(f"bad feature map {v.shape}")
            for j, kk in enumerate(k):
                f["f"][kk] = v[j].astype(np.float16)
                f["lab"][kk] = lab[j]
            f["done"][np.sort(k)] = True
            if b % 50 == 0:
                f.flush()
                log(f"  {model} {dataset}: batch {b}, {int(f['done'][:].sum())} / {n}")
    return out_path


def load_cache(path, augmented=True):
    import h5py
    with h5py.File(path, "r") as f:
        if not f["done"][:].all():
            raise RuntimeError(f"{path} is incomplete; run the extract step again")
        row, ver = f["row"][:], f["ver"][:]
        keep = (ver == 0) | (augmented & (ver > 0))
        if keep.all():
            feat, lab = f["f"][:], f["lab"][:]
        else:
            feat, lab = [], []
            for s in range(0, len(keep), 4096):
                k = keep[s:s + 4096]
                feat.append(f["f"][s:s + 4096][k])
                lab.append(f["lab"][s:s + 4096][k])
            feat, lab = np.concatenate(feat), np.concatenate(lab)
            row, ver = row[keep], ver[keep]
    return row, ver, torch.from_numpy(feat), torch.from_numpy(lab)
