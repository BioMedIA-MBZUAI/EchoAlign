#!/usr/bin/env python3
"""ROC curves and score densities from TEST predictions, as editable SVG (text stays text).

    python scripts/plot_roc.py --config configs/figures_roc.yaml --cohort mimic_iv_echo \
        --predictions results/mimic_iv_echo/probe --labels data/labels/mimic_iv_echo_probe.csv \
        --seed-key mimic/patient_cluster/absolute --out results/mimic_iv_echo/figures
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analysis.metrics import interval, metric_draws, point, cluster_draws, seed_from  # noqa: E402

MM = 1 / 25.4
STYLE = {}


def set_style(style):
    STYLE.update(style)
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.sans-serif": style["font"],
                         "font.size": 7, "text.color": style["ink"], "axes.labelcolor": style["ink"],
                         "axes.edgecolor": style["axis"], "xtick.color": style["axis"], "ytick.color": style["axis"],
                         "xtick.labelcolor": style["ink"], "ytick.labelcolor": style["ink"],
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "axes.grid": True, "grid.color": style["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True})


def auroc_text(y, s, ix, draws):
    lo, hi, _ = interval(metric_draws(y, s, ix, draws, "AUROC"))
    return point(y, s, "AUROC"), lo, hi


def roc_axes(ax, title):
    ax.plot([0, 1], [0, 1], color=STYLE["axis"], lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, fontsize=7, loc="left")


def ordinal_predictions(a, m):
    if m in a.sources:
        d = pd.read_csv(a.sources[m], dtype={"study_id": str})
        return d[d["mode"] == "ordinal"].rename(columns={"y_true": "label"})
    return pd.read_csv(Path(a.predictions) / f"{m}_ordinal_predictions.csv", dtype={"study_id": str})


def cut_points(a, m, task):
    import json
    sel = json.loads((Path(a.predictions) / f"{m}_selection.json").read_text())
    return sel[task]["cuts"]


def grade_colors(ramp, n):
    return [ramp[i] for i in np.round(np.linspace(0, len(ramp) - 1, n)).astype(int)]


def operating_point(y, s, threshold):
    pos = s >= threshold
    return pos[y == 1].mean(), (~pos[y == 0]).mean()


def severity(cfg, a, lab, col, draws, names):
    rows = cfg["severity"]
    m0 = cfg.get("severity_model", cfg["models"][0])
    fig, axes = plt.subplots(len(rows), 3, figsize=(180 * MM, 66 * MM * len(rows)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.45], wspace=0.42, hspace=0.95))
    pred = ordinal_predictions(a, m0)
    rng = np.random.default_rng(0)
    for r, spec in enumerate(rows):
        task = spec["task"]
        per = pred[pred.task == task].set_index("study_id")
        grade, s0 = per.label.to_numpy(int), per.expected_grade.to_numpy(float)
        clusters = lab[(lab.task == task) & lab.split.eq("TEST")].set_index("study_id").cluster_id.loc[per.index]
        ix = col[clusters].to_numpy()
        cuts = cut_points(a, m0, task)
        shades = grade_colors(cfg["grade_shades"], len(spec["grades"]))
        th = [dict(color=shades[k], dash="solid") for k in spec["cuts"].values()]
        for c, ((cut_name, k), st) in enumerate(zip(spec["cuts"].items(), th)):
            ax = axes[r, c]
            y, p = (grade >= k).astype(float), per[f"p_ge_{k}"].to_numpy(float)
            roc_axes(ax, f"{cut_name} ({int(y.sum())} of {len(y)} positive)")
            fpr, tpr, _ = roc_curve(y, p)
            pt, lo, hi = auroc_text(y, p, ix, draws)
            ax.plot(fpr, tpr, color=st["color"], lw=1.5, label=f"AUROC {pt:.3f} ({lo:.3f}-{hi:.3f})")
            sens, spec_ = operating_point(y, s0, cuts[k - 1])
            ax.plot(1 - spec_, sens, "o", ms=5, mfc=st["color"], mec=STYLE["ink"], mew=0.7,
                    label=f"Cut {cuts[k - 1]:.2f}: sens. {sens:.2f}, spec. {spec_:.2f}")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=5.5, handlelength=1.2,
                      borderaxespad=0, labelspacing=0.3)
        axes[r, 0].annotate(spec["title"], (-0.62, 0.5), xycoords="axes fraction", rotation=90, ha="center",
                            va="center", fontsize=7.5, fontweight="bold")
        ax = axes[r, 2]
        grid = np.linspace(-0.25, len(spec["grades"]) - 0.75, 300)
        top = 0
        for g, (gname, shade) in enumerate(zip(spec["grades"], shades)):
            v = s0[grade == g]
            if len(v) < 2 or np.ptp(v) == 0:
                continue
            dens = gaussian_kde(v)(grid)
            top = max(top, dens.max())
            ax.fill_between(grid, dens, color=shade, alpha=0.18, lw=0)
            ax.plot(grid, dens, color=shade, lw=1.3, label=f"{gname} (n = {len(v)})")
        for g, shade in enumerate(shades):
            v = s0[grade == g]
            yj = top * (1.12 + 0.13 * g) + rng.uniform(-0.04, 0.04, len(v)) * top
            ax.scatter(v, yj, s=2.0, color=shade, alpha=0.7, lw=0, rasterized=True)
        ax.set_ylim(0, top * (1.2 + 0.13 * len(spec["grades"])))
        for (cut_name, k), st in zip(spec["cuts"].items(), th):
            ax.axvline(cuts[k - 1], color=st["color"], lw=0.8, ls=(0, (3, 2)))
        ax.set_xlabel(f"{names[m0]} expected grade")
        ax.set_ylabel("Density")
        ax.set_xticks(range(len(spec["grades"])))
        ax.legend(fontsize=5.5, handlelength=1.0, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(spec["grades"]),
                  columnspacing=0.7, borderaxespad=0.2)
        if r == 0:
            axes[r, 2].annotate("Score by true grade (dashed lines: cut points)", (0.5, 1.30), xycoords="axes fraction",
                                ha="center", fontsize=7, fontweight="bold")
    return fig


def roc_band(y, s, ix, draws, grid):
    order = np.argsort(-s, kind="stable")
    y, s, w = y[order], s[order], np.asarray(draws)[:, ix[order]].astype(float)
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(w * y, 1)[:, last]
    fp = np.cumsum(w * (1 - y), 1)[:, last]
    tpr = tp / np.maximum(tp[:, -1:], 1e-12)
    fpr = fp / np.maximum(fp[:, -1:], 1e-12)
    curves = np.stack([np.interp(grid, np.r_[0, f], np.r_[0, t]) for f, t in zip(fpr, tpr)])
    return np.quantile(curves, [0.025, 0.975], axis=0)


def findings(cfg, a, lab, col, draws, names):
    spec = cfg["findings"]
    m = spec["model"]
    pred = ordinal_predictions(a, m)
    tasks = list(spec["tasks"].items())
    ncol = 2
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(120 * MM, 60 * MM * nrow), squeeze=False, sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.12, hspace=0.28))
    ink, soft = STYLE["ink"], STYLE["band"]
    grid = np.linspace(0, 1, 201)
    for i, (task, t) in enumerate(tasks):
        ax = axes[i // ncol, i % ncol]
        per = pred[pred.task == task].set_index("study_id")
        g, s0 = per.label.to_numpy(int), per.expected_grade.to_numpy(float)
        y, p = (g >= t["cut"]).astype(float), per[f"p_ge_{t['cut']}"].to_numpy(float)
        clusters = lab[(lab.task == task) & lab.split.eq("TEST")].set_index("study_id").cluster_id.loc[per.index]
        ix = col[clusters].to_numpy()
        pt, lo, hi = auroc_text(y, p, ix, draws)
        lo_b, hi_b = roc_band(y, p, ix, draws, grid)
        fpr, tpr, _ = roc_curve(y, p)
        sens, spec_ = operating_point(y, s0, cut_points(a, m, task)[t["cut"] - 1])
        ax.plot([0, 1], [0, 1], color=STYLE["axis"], lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.fill_between(grid, lo_b, hi_b, color=soft, lw=0, zorder=2)
        ax.plot(fpr, tpr, color=ink, lw=1.3, zorder=3)
        ax.plot([1 - spec_, 1 - spec_], [0, sens], color=ink, lw=0.5, ls=(0, (1, 1.5)), zorder=3)
        ax.plot([0, 1 - spec_], [sens, sens], color=ink, lw=0.5, ls=(0, (1, 1.5)), zorder=3)
        ax.plot(1 - spec_, sens, "o", ms=5, mfc=ink, mec="white", mew=0.8, zorder=4)
        box = dict(boxstyle="square,pad=0.15", fc="white", ec="none")
        ax.annotate(f"Sens. {sens:.2f}\nSpec. {spec_:.2f}", (1 - spec_, sens), xytext=(7, -6), textcoords="offset points",
                    fontsize=5.5, color=ink, va="top", bbox=box, zorder=5)
        ax.text(0.96, 0.30, f"AUROC {pt:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                color=ink, bbox=box, zorder=5)
        ax.text(0.96, 0.29, f"95% CI {lo:.3f}-{hi:.3f}\n{int(y.sum())} of {len(y)} studies positive",
                transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color=STYLE["axis"], bbox=box, zorder=5,
                linespacing=1.4)
        ax.set_title(t["title"], fontsize=7, loc="left", fontweight="bold", pad=4)
        ax.set(xlim=(0, 1), ylim=(0, 1.01), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], aspect="equal")
        if i // ncol == nrow - 1:
            ax.set_xlabel("1 - specificity")
        if i % ncol == 0:
            ax.set_ylabel("Sensitivity")
    for j in range(len(tasks), nrow * ncol):
        axes[j // ncol, j % ncol].set_visible(False)
    fig.subplots_adjust(top=0.9)
    fig.suptitle(f"{names[m]}, {spec['cohort_title']}: {spec['cut_title']}", fontsize=7.5, x=0.5, y=0.985)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/figures_roc.yaml"))
    ap.add_argument("--cohort", required=True, help="cohort in --config")
    ap.add_argument("--predictions", required=True, help="folder with the probe outputs of scripts/probe.py")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--seed-key", required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--source", action="append", default=[],
                    help="MODEL=CSV: read this model's grades from a scripts/probe_echoprime_sections.py prediction file")
    ap.add_argument("--font-dir", help="folder of .ttf files to register (for example Liberation Sans, metric-compatible "
                                       "with Arial, where Helvetica and Arial are not installed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.sources = dict(x.split("=", 1) for x in a.source)
    top = yaml.safe_load(open(a.config))
    cfg = dict(top[a.cohort], models=top["models"], grade_shades=top["grade_shades"])
    style = dict(top["style"])
    if a.font_dir:
        from matplotlib import font_manager
        names = []
        for f in sorted(Path(a.font_dir).glob("*.ttf")):
            font_manager.fontManager.addfont(str(f))
            names.append(font_manager.FontProperties(fname=str(f)).get_name())
        style["font"] = list(dict.fromkeys(style["font"][:2] + names + style["font"][2:]))
    set_style(style)
    names = {m: v["name"] for m, v in top["model_style"].items()}
    lab = pd.read_csv(a.labels, dtype={"study_id": str, "task": str, "cluster_id": str})
    if "cluster_id" not in lab:
        lab["cluster_id"] = lab.study_id
    clusters = np.array(sorted(lab[lab.split.eq("TEST")].cluster_id.unique()))
    col = pd.Series(np.arange(len(clusters)), index=clusters)
    draws = cluster_draws(len(clusters), a.replicates, seed_from(a.seed_key))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, build in (("roc_severity", severity), ("roc_findings", findings)):
        if name.split("_")[1] not in cfg:
            continue
        fig = build(cfg, a, lab, col, draws, names)
        for ext in ("svg", "pdf"):
            fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)
        print(f"{name} -> {out}")


if __name__ == "__main__":
    main()
