import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import SEED, echoprime_crop_box
from .metrics import dice_iou, fg_mean

LR_GRID = {"linear": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1], "decoder": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]}
EPOCHS = {"linear": 30, "decoder": 60}
BATCH, WEIGHT_DECAY = 32, 1e-4


class LinearHead(nn.Module):
    def __init__(self, c_in, n_cls):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.fc = nn.Conv2d(c_in, n_cls, 1)

    def forward(self, x):
        return F.interpolate(self.fc(self.bn(x)), size=(224, 224), mode="bilinear", align_corners=False)


class DecoderHead(nn.Module):
    def __init__(self, c_in, n_cls):
        super().__init__()
        self.proj = nn.Conv2d(c_in, 256, 1)
        self.b1 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.cls = nn.Conv2d(256, n_cls, 1)

    def forward(self, x):
        x = F.interpolate(self.proj(x), size=(56, 56), mode="bilinear", align_corners=False)
        x = F.interpolate(self.b1(x), scale_factor=2, mode="bilinear", align_corners=False)
        x = F.interpolate(self.b2(x), scale_factor=2, mode="bilinear", align_corners=False)
        return self.cls(x)


def make_head(kind, c_in, n_cls):
    return LinearHead(c_in, n_cls) if kind == "linear" else DecoderHead(c_in, n_cls)


def n_params(head):
    return sum(p.numel() for p in head.parameters())


def to_native(logits224, model, shape):
    h, w = shape
    if model != "echoprime":
        return F.interpolate(logits224[None].float(), size=(h, w), mode="bilinear", align_corners=False)[0].argmax(0).to(torch.uint8)
    y0, y1, x0, x1 = echoprime_crop_box(h, w)
    out = torch.zeros((h, w), dtype=torch.uint8, device=logits224.device)
    out[y0:y1, x0:x1] = F.interpolate(logits224[None].float(), size=(y1 - y0, x1 - x0), mode="bilinear",
                                      align_corners=False)[0].argmax(0).to(torch.uint8)
    return out


def _autocast(kind, device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=kind == "decoder" and device.type == "cuda")


@torch.no_grad()
def predict(head, kind, model, feat, idx, shapes, device):
    head.eval()
    out = []
    for s in range(0, len(idx), 64):
        x = feat[idx[s:s + 64]].to(device, non_blocking=True).float()
        with _autocast(kind, device):
            lg = head(x).float()
        out += [to_native(lg[j], model, shapes[s + j]) for j in range(lg.shape[0])]
    return out


def score(preds, gts, n_cls):
    D, J = [], []
    for p, t in zip(preds, gts):
        d, j = dice_iou(p, t.to(p.device), n_cls)
        D.append([d[c] for c in range(n_cls)])
        J.append([j[c] for c in range(n_cls)])
    return np.array(D, float), np.array(J, float)


def fit(kind, model, feat, lab, row, ver, split, val_masks, val_shapes, n_cls, seed=0, lr_grid=None, epochs=None,
        batch_size=BATCH, weight_decay=WEIGHT_DECAY, base_seed=SEED, device="cuda", log=print):
    device = torch.device(device)
    lr_grid = lr_grid or LR_GRID[kind]
    epochs = epochs or EPOCHS[kind]
    split = np.asarray(split)
    tr_rows = np.unique(row[split == "TRAIN"])
    tr_idx = {r: np.flatnonzero(row == r) for r in tr_rows}
    va = np.flatnonzero((split == "VAL") & (ver == 0))
    c_in = feat.shape[1]
    curve, best = [], dict(val=-1.0)
    for lr in lr_grid:
        torch.manual_seed(base_seed + seed)
        rng = np.random.default_rng([base_seed, seed, int(lr * 1e6)])
        head = make_head(kind, c_in, n_cls).to(device)
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        for ep in range(1, epochs + 1):
            head.train()
            order = rng.permutation(tr_rows)
            pick = np.array([tr_idx[r][rng.integers(0, len(tr_idx[r]))] if kind == "decoder" else tr_idx[r][0] for r in order])
            if kind == "linear":
                assert (ver[pick] == 0).all()
            tot = 0.0
            for s in range(0, len(pick), batch_size):
                b = pick[s:s + batch_size]
                if len(b) < 2:                                          # BatchNorm needs two samples
                    continue
                x = feat[b].to(device, non_blocking=True).float()
                y = lab[b].to(device, non_blocking=True).long()
                with _autocast(kind, device):
                    lg = head(x)
                lg = lg.float()
                loss = F.cross_entropy(lg, y)
                if kind == "decoder":
                    p = lg.softmax(1)[:, 1:]
                    oh = F.one_hot(y, n_cls).permute(0, 3, 1, 2)[:, 1:].float()
                    inter = (p * oh).sum((0, 2, 3))
                    den = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
                    loss = loss + (1 - (2 * inter + 1) / (den + 1)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            Dv, _ = score(predict(head, kind, model, feat, va, val_shapes, device), val_masks, n_cls)
            v = fg_mean(Dv)
            curve.append(dict(lr=lr, epoch=ep, train_loss=tot / len(pick), val_fg_dice=v))
            log(f"    lr {lr:g} epoch {ep} loss {tot / len(pick):.4f} val {v:.4f}")
            if v > best["val"]:
                best = dict(val=v, lr=lr, epoch=ep, state=copy.deepcopy(head.state_dict()))
    if "state" not in best:
        raise RuntimeError(f"{kind} {model}: no finite VAL Dice for any learning rate")
    head = make_head(kind, c_in, n_cls).to(device)
    head.load_state_dict(best["state"])
    return head, dict(chosen_lr=best["lr"], chosen_epoch=best["epoch"], val_fg_dice=best["val"], lr_grid=list(lr_grid),
                      epochs=epochs, batch=batch_size, weight_decay=weight_decay, params=n_params(head), in_channels=c_in,
                      edge_lr=best["lr"] in (lr_grid[0], lr_grid[-1]), last_epoch=best["epoch"] == epochs, curve=curve)
