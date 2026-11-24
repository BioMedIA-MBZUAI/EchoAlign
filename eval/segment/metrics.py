import numpy as np
import pandas as pd

from eval.analysis.metrics import cluster_draws, interval
from .features import SEED

N_BOOT = 1000


def dice_iou(pred, gt, n_cls):
    d, j = {}, {}
    for c in range(n_cls):
        p, t = pred == c, gt == c
        inter, sp, st = int((p & t).sum()), int(p.sum()), int(t.sum())
        d[c] = 2 * inter / (sp + st) if sp + st else np.nan
        j[c] = inter / (sp + st - inter) if sp + st else np.nan
    return d, j


def fg_mean(D):
    return float(np.nanmean(np.nanmean(D[:, 1:], 0)))


def image_rows(D, J, meta, classes):
    out = meta.reset_index(drop=True).copy()
    for c, name in enumerate(classes):
        out[f"dice_{name}"] = D[:, c]
        out[f"iou_{name}"] = J[:, c]
    return out


def patient_table(per_seed, classes):
    cols = [f"{k}_{c}" for k in ("dice", "iou") for c in classes]
    tabs = [d.groupby("patient")[cols].mean().sort_index() for d in per_seed]
    return (pd.concat(tabs).groupby(level=0).mean() if len(tabs) > 1 else tabs[0]), tabs


def values(P, classes):
    dm = P[[f"dice_{c}" for c in classes]].mean(0, skipna=True).to_numpy()
    im = P[[f"iou_{c}" for c in classes]].mean(0, skipna=True).to_numpy()
    out = {f"dice_{c}": dm[i] for i, c in enumerate(classes)}
    out.update(fg_mean=float(np.nanmean(dm[1:])), miou_fg=float(np.nanmean(im[1:])), miou_all=float(np.nanmean(im)))
    return out


def patient_draws(n_patients, n_boot=N_BOOT, seed=SEED):
    return [np.repeat(np.arange(n_patients), c) for c in cluster_draws(n_patients, n_boot, seed)]


def boot_values(P, classes, draws):
    X, cols = P.to_numpy(), list(P.columns)
    return pd.DataFrame([values(pd.DataFrame(X[d], columns=cols), classes) for d in draws])


def ci(v):
    return interval(v)[:2]


def native_tables(runs, classes_of, grid_of, reference="echoalign", n_boot=N_BOOT, seed=SEED):
    nat, pair, boots, vals = [], [], {}, {}
    datasets = list(dict.fromkeys(k[0] for k in runs))
    for ds in datasets:
        classes = classes_of[ds]
        keys = [k for k in runs if k[0] == ds]
        pts = draws = None
        for k in keys:
            P, per = patient_table(runs[k], classes)
            if pts is None:
                pts, draws = list(P.index), patient_draws(len(P), n_boot, seed)
            if list(P.index) != pts:
                raise RuntimeError(f"{k}: patients differ from the other runs of {ds}")
            v, B = values(P, classes), boot_values(P, classes, draws)
            boots[k], vals[k] = B, v
            ps = [values(t, classes) for t in per]
            for m in [f"dice_{c}" for c in classes[1:]] + ["fg_mean", "miou_fg", "miou_all"]:
                lo, hi = ci(B[m])
                nat.append(dict(dataset=ds, model=k[1], head=k[2], grid=f"{grid_of[k[1]]}x{grid_of[k[1]]}", metric=m,
                                value=v[m], lo=lo, hi=hi, seed_min=min(p[m] for p in ps), seed_max=max(p[m] for p in ps),
                                n_seeds=len(ps), n_patients=len(pts)))
        for (_, model, head) in keys:
            ref = (ds, reference, head)
            if model == reference or ref not in boots:
                continue
            for m in [f"dice_{c}" for c in classes[1:]] + ["fg_mean", "miou_fg"]:
                lo, hi = ci(boots[ref][m].to_numpy() - boots[(ds, model, head)][m].to_numpy())
                pair.append(dict(dataset=ds, head=head, comparison=f"{reference} - {model}", metric=m,
                                 diff=vals[ref][m] - vals[(ds, model, head)][m], lo=lo, hi=hi))
    cross = []
    if len(datasets) > 1:
        for head in dict.fromkeys(k[2] for k in runs):
            models = [m for m in dict.fromkeys(k[1] for k in runs) if all((d, m, head) in boots for d in datasets)]
            mean_b = {m: np.mean([boots[(d, m, head)]["fg_mean"].to_numpy() for d in datasets], 0) for m in models}
            mean_v = {m: float(np.mean([vals[(d, m, head)]["fg_mean"] for d in datasets])) for m in models}
            for m in models:
                cross.append(dict(head=head, model=m, comparison="", datasets="+".join(datasets), value=mean_v[m],
                                  lo=ci(mean_b[m])[0], hi=ci(mean_b[m])[1]))
            if reference in models:
                for m in models:
                    if m != reference:
                        d = mean_b[reference] - mean_b[m]
                        cross.append(dict(head=head, model=m, comparison=f"{reference} - {m}", datasets="+".join(datasets),
                                          value=mean_v[reference] - mean_v[m], lo=ci(d)[0], hi=ci(d)[1]))
    return pd.DataFrame(nat), pd.DataFrame(pair), pd.DataFrame(cross)


def transfer_table(runs, grid_of, in_domain=None, reference="echoalign", n_boot=N_BOOT, seed=SEED):
    cls = ["background", "LV_cavity"]
    rows, pair = [], []
    for tgt in dict.fromkeys(k[0] for k in runs):
        keys = [k for k in runs if k[0] == tgt]
        pts = draws = None
        tb = {}
        for k in keys:
            P, per = patient_table(runs[k], cls)
            if pts is None:
                pts, draws = list(P.index), patient_draws(len(P), n_boot, seed)
            if list(P.index) != pts:
                raise RuntimeError(f"{k}: patients differ from the other runs of {tgt}")
            B = boot_values(P, cls, draws)["dice_LV_cavity"].to_numpy()
            v = values(P, cls)["dice_LV_cavity"]
            tb[k] = (v, B)
            ps = [values(t, cls)["dice_LV_cavity"] for t in per]
            indom = None
            if in_domain is not None and len(in_domain):
                q = in_domain[(in_domain.dataset == tgt) & (in_domain.model == k[1]) & (in_domain.head == k[2])
                              & (in_domain.metric == "dice_LV_cavity")]
                indom = float(q.value.iloc[0]) if len(q) else None
            rows.append(dict(target=tgt, model=k[1], head=k[2], grid=f"{grid_of[k[1]]}x{grid_of[k[1]]}", lv_dice=v,
                             lo=ci(B)[0], hi=ci(B)[1], seed_min=min(ps), seed_max=max(ps), in_domain_lv_dice=indom,
                             transfer_minus_in_domain=None if indom is None else v - indom, n_seeds=len(ps), n_patients=len(pts)))
        for (_, model, head) in keys:
            ref = (tgt, reference, head)
            if model != reference and ref in tb:
                d = tb[ref][1] - tb[(tgt, model, head)][1]
                pair.append(dict(target=tgt, head=head, comparison=f"{reference} - {model}",
                                 diff=tb[ref][0] - tb[(tgt, model, head)][0], lo=ci(d)[0], hi=ci(d)[1]))
    return pd.DataFrame(rows), pd.DataFrame(pair)
