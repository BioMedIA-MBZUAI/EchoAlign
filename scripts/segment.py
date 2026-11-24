#!/usr/bin/env python3
"""Per-pixel segmentation with frozen encoders: the paper's evaluation, from raw data to tables.

    python scripts/segment.py                                  # every dataset, model, head, the transfer task
    python scripts/segment.py --tasks camus --models echoalign panecho --heads linear
    python scripts/segment.py --steps report                   # tables from finished runs
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from eval.analysis.data import datasets_config  # noqa: E402
from eval.embed import METHODS as EMBED_METHODS  # noqa: E402
from eval.segment import data as D  # noqa: E402
from eval.segment import features as FE  # noqa: E402
from eval.segment import heads as H  # noqa: E402
from eval.segment import metrics as M  # noqa: E402

TASKS = list(D.DATASETS) + ["transfer"]
STEPS = ["prep", "extract", "train", "report"]
DECODER_SEEDS = {"camus": [0, 1, 2], "transfer": [0, 1, 2], "echonet_dynamic": [0, 1, 2], "echonet_pediatric": [0, 1, 2]}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    ap.add_argument("--models", nargs="+", choices=EMBED_METHODS, default=list(FE.MODELS),
                    help=f"models with segmentation support: {', '.join(FE.MODELS)}; others are skipped")
    ap.add_argument("--heads", nargs="+", choices=["linear", "decoder"], default=["linear", "decoder"])
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    ap.add_argument("--transfer-targets", nargs="+", choices=["echonet_dynamic", "echonet_pediatric"],
                    default=["echonet_dynamic", "echonet_pediatric"])
    ap.add_argument("--config", default=str(datasets_config()), help="dataset folders")
    ap.add_argument("--out", default=str(ROOT / "results/segmentation"))
    ap.add_argument("--cache-dir", default=None, help="feature caches (default <out>/cache)")
    ap.add_argument("--weights", default=str(ROOT / "weights/EchoAlign.pt"))
    ap.add_argument("--echoprime-repo", default=str(ROOT / "models/EchoPrime"),
                    help="EchoPrime clone; weights from model_data/weights/echo_prime_encoder.pt")
    ap.add_argument("--echoprime-weights", default=None, help="EchoPrime video encoder weights (overrides the repo path)")
    ap.add_argument("--panecho-source", default=str(ROOT / "models/PanEcho"), help="PanEcho clone or GitHub repo")
    ap.add_argument("--panecho-github", action="store_true", help="load PanEcho from GitHub instead of the local clone")
    ap.add_argument("--echoclip-model-id", default="hf-hub:mkaichristensen/echo-clip")
    ap.add_argument("--pediatric-frame-base", type=int, choices=[0, 1], default=1,
                    help="EchoNet-Pediatric VolumeTracings Frame numbering (1: every Frame minus 1)")
    ap.add_argument("--n-aug", type=int, default=4, help="fixed augmented copies per TRAIN frame for the decoder")
    ap.add_argument("--linear-lrs", type=float, nargs="+", default=H.LR_GRID["linear"])
    ap.add_argument("--decoder-lrs", type=float, nargs="+", default=H.LR_GRID["decoder"])
    ap.add_argument("--linear-epochs", type=int, default=H.EPOCHS["linear"])
    ap.add_argument("--decoder-epochs", type=int, default=H.EPOCHS["decoder"])
    ap.add_argument("--linear-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--decoder-seeds", type=int, nargs="+", default=None,
                    help="decoder seeds of every task (default: DECODER_SEEDS, 0 1 2 for every task)")
    ap.add_argument("--batch-size", type=int, default=H.BATCH)
    ap.add_argument("--weight-decay", type=float, default=H.WEIGHT_DECAY)
    ap.add_argument("--seed", type=int, default=FE.SEED, help="base seed: augmentation, head init (+ seed), bootstrap")
    ap.add_argument("--n-boot", type=int, default=M.N_BOOT)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extract-batch-size", type=int, default=None, help="default per model: 16, 4, 32, 32")
    ap.add_argument("--save-predictions", action="store_true", help="also save TEST class maps (.npz)")
    ap.add_argument("--limit", type=int, default=None, help="first N patients per split (smoke test; use a separate --out)")
    ap.add_argument("--disable-cudnn", action="store_true",
                    help="turn off cuDNN/MIOpen")
    a = ap.parse_args()
    for m in a.models:
        if m not in FE.MODELS:
            print(f"skip {m}: no segmentation support", flush=True)
    a.models = [m for m in a.models if m in FE.MODELS]
    return a


def seeds_for(a, task):
    return {"linear": a.linear_seeds, "decoder": a.decoder_seeds or DECODER_SEEDS[task]}


def datasets_needed(a):
    ds = [t for t in a.tasks if t != "transfer"]
    if "transfer" in a.tasks:
        ds += ["camus"] + a.transfer_targets
    return [d for d in D.DATASETS if d in ds]


def fit_datasets(a):
    return {t if t != "transfer" else "camus" for t in a.tasks}


def step_prep(a, prep):
    roots = D.dataset_roots(a.config)
    for ds in datasets_needed(a):
        if (prep / f"{ds}_manifest.csv").exists():
            log("prep: exists", ds)
            continue
        if roots.get(ds) is None:
            raise SystemExit(f"set the {ds} folder in {a.config}")
        log("prep:", ds, roots[ds])
        D.prepare(ds, roots[ds], prep, workers=a.workers, pediatric_frame_base=a.pediatric_frame_base, limit=a.limit)
        log("prep:", (prep / f"{ds}_summary.json").read_text())


def cache_path(a, ds, model):
    return Path(a.cache_dir or Path(a.out) / "cache") / ds / f"{model}.h5"


def encoder_kwargs(a):
    return dict(device=a.device, weights=a.weights, echoprime_repo=a.echoprime_repo, echoprime_weights=a.echoprime_weights,
                panecho_source=a.panecho_source, panecho_local=not a.panecho_github, echoclip_model_id=a.echoclip_model_id)


def step_extract(a, prep):
    import h5py
    fits = fit_datasets(a)
    for ds in datasets_needed(a):
        table = D.image_table(D.load_manifest(prep, ds))
        n_aug = a.n_aug if ("decoder" in a.heads and ds in fits) else 0
        for model in a.models:
            p = cache_path(a, ds, model)
            if p.exists():
                with h5py.File(p, "r") as f:
                    n_aug = max(n_aug, int(f.attrs.get("n_aug", 0)))
            FE.extract(model, ds, table, prep, p, n_aug=n_aug, seed=a.seed, batch_size=a.extract_batch_size,
                       workers=a.workers, log=log, **encoder_kwargs(a))


def _masks(prep, ds, table, rows, binary):
    out = []
    for r in rows:
        m = D.native_mask(prep, ds, table["clip"].iloc[r], table.phase.iloc[r])
        out.append(torch.from_numpy((m == 1).astype(np.uint8) if binary else m.astype(np.uint8)))
    return out


def step_train(a, prep):
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    lrs = {"linear": a.linear_lrs, "decoder": a.decoder_lrs}
    epochs = {"linear": a.linear_epochs, "decoder": a.decoder_epochs}
    for task in a.tasks:
        src = "camus" if task == "transfer" else task
        seeds = seeds_for(a, task)
        binary = task == "transfer"
        n_cls = 2 if binary else len(D.CLASSES[src])
        classes = ["background", "LV_cavity"] if binary else D.CLASSES[src]
        targets = ["camus"] + a.transfer_targets if binary else [src]
        table = D.image_table(D.load_manifest(prep, src))
        for model in a.models:
            todo = [(h, s) for h in a.heads for s in seeds[h]
                    if not (Path(a.out) / "runs" / task / f"{model}_{h}_s{s}.json").exists()]
            if not todo:
                log("train: done", task, model)
                continue
            for head in dict.fromkeys(h for h, _ in todo):
                row, ver, feat, lab = FE.load_cache(cache_path(a, src, model), augmented=head == "decoder")
                if binary:
                    lab = (lab == 1).to(torch.uint8)
                split = table.split.to_numpy()[row]
                va = np.flatnonzero((split == "VAL") & (ver == 0))
                va_shapes = [(int(table.H.iloc[r]), int(table.W.iloc[r])) for r in row[va]]
                va_gt = _masks(prep, src, table, row[va], binary)
                for h, seed in todo:
                    if h != head:
                        continue
                    log("train:", task, model, head, "seed", seed, "items", len(row), "VAL images", len(va))
                    t0 = time.time()
                    net, info = H.fit(head, model, feat, lab, row, ver, split, va_gt, va_shapes, n_cls, seed=seed,
                                      lr_grid=lrs[head], epochs=epochs[head], batch_size=a.batch_size,
                                      weight_decay=a.weight_decay, base_seed=a.seed, device=device, log=log)
                    test(a, prep, task, model, head, seed, net, info, targets, src, classes, n_cls, binary,
                         (table, row, ver, feat), device, t0)
                del feat, lab


def test(a, prep, task, model, head, seed, net, info, targets, src, classes, n_cls, binary, src_cache, device, t0):
    od = Path(a.out) / "runs" / task
    od.mkdir(parents=True, exist_ok=True)
    name = f"{model}_{head}_s{seed}"
    recs, preds = [], {}
    for tgt in targets:
        if tgt == src:
            table, row, ver, feat = src_cache
        else:
            table = D.image_table(D.load_manifest(prep, tgt))
            row, ver, feat, _ = FE.load_cache(cache_path(a, tgt, model), augmented=False)
        te = np.flatnonzero((table.split.to_numpy()[row] == "TEST") & (ver == 0))
        shapes = [(int(table.H.iloc[r]), int(table.W.iloc[r])) for r in row[te]]
        pr = H.predict(net, head, model, feat, te, shapes, device)
        Dm, Jm = H.score(pr, _masks(prep, tgt, table, row[te], binary), n_cls)
        meta = table.iloc[row[te]][["image", "patient", "view", "phase"]].assign(target=tgt)
        recs.append(M.image_rows(Dm, Jm, meta[["target", "image", "patient", "view", "phase"]], classes))
        if a.save_predictions:
            preds.update({f"{tgt}__{img}": p.cpu().numpy() for img, p in zip(meta.image, pr)})
        log(f"  TEST {tgt}: image-level foreground Dice {M.fg_mean(Dm):.4f}")
    pd.concat(recs).to_csv(od / f"{name}.csv", index=False)
    if a.save_predictions:
        np.savez_compressed(od / f"{name}.npz", **preds)
    info.update(task=task, model=model, head=head, seed=seed, n_aug=int(src_cache[2].max()) if head == "decoder" else 0,
                grid=FE.GRID[model], elapsed_seconds=time.time() - t0)
    (od / f"{name}.json").write_text(json.dumps(info, indent=1) + "\n")
    log("  done", task, name, "lr", info["chosen_lr"], "epoch", info["chosen_epoch"], f"VAL {info['val_fg_dice']:.4f}")


def step_report(a):
    runs_dir, td = Path(a.out) / "runs", Path(a.out) / "tables"
    td.mkdir(parents=True, exist_ok=True)
    native, transfer, sel, par = {}, {}, [], []
    for task in a.tasks:
        seeds = seeds_for(a, task)
        for model in a.models:
            for head in a.heads:
                files = [runs_dir / task / f"{model}_{head}_s{s}" for s in seeds[head]]
                if not all(f.with_suffix(".json").exists() for f in files):
                    log("report: missing", task, model, head)
                    continue
                per = [pd.read_csv(f.with_suffix(".csv"), dtype={"patient": str}) for f in files]
                for s, f in zip(seeds[head], files):
                    j = json.loads(f.with_suffix(".json").read_text())
                    sel.append(dict(task=task, model=model, head=head, seed=s, lr=j["chosen_lr"], epoch=j["chosen_epoch"],
                                    val_fg_dice=j["val_fg_dice"], edge_lr=j["edge_lr"], last_epoch=j["last_epoch"]))
                    if s == seeds[head][0] and task != "transfer":
                        par.append(dict(task=task, model=model, head=head, in_channels=j["in_channels"], grid=j["grid"],
                                        params_total=j["params"]))
                if task == "transfer":
                    for tgt in dict.fromkeys(per[0].target):
                        transfer[(tgt, model, head)] = [p[p.target == tgt] for p in per]
                else:
                    native[(task, model, head)] = per
    nat = pair = cross = tr = trp = pd.DataFrame()
    if native:
        nat, pair, cross = M.native_tables(native, D.CLASSES, FE.GRID, n_boot=a.n_boot, seed=a.seed)
    if transfer:
        tr, trp = M.transfer_table(transfer, FE.GRID, nat, n_boot=a.n_boot, seed=a.seed)
    for name, df in (("native", nat), ("paired", pair), ("cross_dataset", cross), ("transfer", tr),
                     ("transfer_paired", trp), ("selection", pd.DataFrame(sel)), ("params", pd.DataFrame(par))):
        df.to_csv(td / f"segmentation_{name}.csv", index=False)
    if len(nat):
        q = nat[nat.metric == "fg_mean"]
        log("TEST foreground Dice (95% interval):\n" + q[["dataset", "model", "head", "grid", "value", "lo", "hi"]].to_string(index=False))
    if len(tr):
        log("Transfer, LV cavity Dice:\n" + tr[["target", "model", "head", "lv_dice", "lo", "hi"]].to_string(index=False))
    log("tables:", td)


def main():
    a = parse()
    if a.disable_cudnn:
        torch.backends.cudnn.enabled = False
    prep = Path(a.out) / "prep"
    log("segment:", " ".join(sys.argv[1:]))
    if "prep" in a.steps:
        step_prep(a, prep)
    if "extract" in a.steps:
        step_extract(a, prep)
    if "train" in a.steps:
        step_train(a, prep)
    if "report" in a.steps:
        step_report(a)


if __name__ == "__main__":
    main()
