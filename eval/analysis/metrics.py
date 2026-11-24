import hashlib

import numpy as np


def seed_from(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def cluster_draws(n_clusters, replicates, seed):
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(n_clusters, size=n_clusters), minlength=n_clusters)
                     for _ in range(replicates)])


def metric_draws(y_true, y_pred, cluster_index, draws, metric, n_classes=None):
    y = np.asarray(y_true, float)
    p = np.asarray(y_pred, float)
    ix = np.asarray(cluster_index)
    draws = np.asarray(draws)
    if metric == "QWK":
        from .ordinal import qwk
        return qwk(y.astype(int), p.astype(int), n_classes, draws[:, ix].astype(float))
    if metric == "RMSE":
        w = draws[:, ix].astype(float)
        den = w.sum(1)
        return np.sqrt(np.divide(w @ ((y - p) ** 2), den, out=np.full(len(draws), np.nan), where=den > 0))
    if metric != "AUROC":
        raise ValueError("metric must be AUROC, RMSE or QWK")
    order = np.argsort(p, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    w = draws[:, ix[order]].astype(float)
    pos = np.add.reduceat(w * y[order], starts, axis=1)
    neg = np.add.reduceat(w * (1 - y[order]), starts, axis=1)
    num = np.sum(pos * (np.cumsum(neg, axis=1) - 0.5 * neg), axis=1)
    den = pos.sum(1) * neg.sum(1)
    return np.divide(num, den, out=np.full(len(draws), np.nan), where=den > 0)


def point(y_true, y_pred, metric, n_classes=None):
    y = np.asarray(y_true, float)
    return float(metric_draws(y, y_pred, np.arange(len(y)), np.ones((1, len(y)), int), metric, n_classes)[0])


def interval(values):
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if not ok.any():
        return np.nan, np.nan, 0
    lo, hi = np.quantile(v[ok], [0.025, 0.975])
    return float(lo), float(hi), int(ok.sum())


def auroc_ci(y_true, y_score, cluster_index, draws):
    lo, hi, valid = interval(metric_draws(y_true, y_score, cluster_index, draws, "AUROC"))
    return dict(auroc=point(y_true, y_score, "AUROC"), lower=lo, upper=hi, valid_replicates=valid)


def rmse_ci(y_true, y_pred, cluster_index, draws):
    lo, hi, valid = interval(metric_draws(y_true, y_pred, cluster_index, draws, "RMSE"))
    return dict(rmse=point(y_true, y_pred, "RMSE"), lower=lo, upper=hi, valid_replicates=valid)
