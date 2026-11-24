import numpy as np
from sklearn.preprocessing import StandardScaler

from .metrics import auroc_ci, interval
from .probes import MIN_VAL, fit_logistic, select_logistic


def expected_grade(probs):
    return np.asarray(probs, float) @ np.arange(probs.shape[1], dtype=float)


def discretize(r, cuts):
    return np.searchsorted(np.asarray(cuts, float), np.asarray(r, float), side="right")


def qwk(y_true, y_pred, n_classes, weights=None):
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    w = np.ones((1, len(y_true))) if weights is None else np.asarray(weights, float)
    onehot = np.zeros((len(y_true), n_classes * n_classes))
    onehot[np.arange(len(y_true)), y_true * n_classes + y_pred] = 1
    O = (w @ onehot).reshape(len(w), n_classes, n_classes)
    n = O.sum((1, 2))
    i, j = np.meshgrid(np.arange(n_classes), np.arange(n_classes), indexing="ij")
    d = (i - j) ** 2
    E = O.sum(2)[:, :, None] * O.sum(1)[:, None, :] / np.maximum(n, 1e-12)[:, None, None]
    den = (d * E).sum((1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(den > 0, 1 - (d * O).sum((1, 2)) / den, np.nan)
    return float(k[0]) if weights is None else k


def fit_cuts(r, y, n_classes):
    grid = np.quantile(r, np.linspace(0.01, 0.99, 99))
    cuts = []
    for k in range(1, n_classes):
        best, best_q = None, -np.inf
        for c in grid:
            if cuts and c <= cuts[-1]:
                continue
            trial = cuts + [c] + list(np.linspace(c + 1e-6, n_classes - 1, n_classes - 1 - k))
            q = qwk(y, discretize(r, sorted(trial)), n_classes)
            if q > best_q:
                best_q, best = q, c
        cuts.append(best if best is not None else k - 0.5)
    return sorted(cuts)


def ordinal_probe(x_train, y_train, x_val, y_val, x_test, n_classes, task=""):
    y_train, y_val = np.asarray(y_train, int), np.asarray(y_val, int)
    for split, y in (("TRAIN", y_train), ("VAL", y_val)):
        missing = sorted(set(range(n_classes)) - set(y.tolist()))
        if missing:
            raise ValueError(f"{task}: grades {missing} missing from {split}")
    sc = StandardScaler().fit(np.asarray(x_train, np.float64))
    tr, va = sc.transform(np.asarray(x_train, np.float64)), sc.transform(np.asarray(x_val, np.float64))

    def val_qwk(m):
        cuts = fit_cuts(expected_grade(m.predict_proba(tr)), y_train, n_classes)
        return qwk(y_val, discretize(expected_grade(m.predict_proba(va)), cuts), n_classes)

    usable = len(y_val) >= MIN_VAL and len(np.unique(y_val)) >= 2
    best, best_qwk, curve = select_logistic(tr, y_train, val_qwk, usable, "validation_qwk")
    x_fit = np.concatenate([x_train, x_val]).astype(np.float64)
    y_fit = np.concatenate([y_train, y_val])
    sc = StandardScaler().fit(x_fit)
    head, retried = fit_logistic(best, sc.transform(x_fit), y_fit)
    cuts = fit_cuts(expected_grade(head.predict_proba(sc.transform(x_fit))), y_fit, n_classes)
    probs = head.predict_proba(sc.transform(np.asarray(x_test, np.float64)))
    grades = discretize(expected_grade(probs), cuts)
    return probs, grades, dict(C=best, C_source="validation" if usable else "default", validation_qwk=best_qwk,
                               cuts=[float(c) for c in cuts], converged=True, final_retry_10000=retried, curve=curve)


def qwk_ci(y_true, y_pred, n_classes, cluster_index, draws):
    lo, hi, valid = interval(qwk(y_true, y_pred, n_classes, np.asarray(draws)[:, np.asarray(cluster_index)]))
    return dict(qwk=qwk(y_true, y_pred, n_classes), lower=lo, upper=hi, valid_replicates=valid)


def boundary_aurocs(y_true, probs, cluster_index, draws):
    y_true = np.asarray(y_true, int)
    return [dict(boundary=k, **auroc_ci((y_true >= k).astype(int), probs[:, k:].sum(1), cluster_index, draws))
            for k in range(1, probs.shape[1]) if 0 < (y_true >= k).sum() < len(y_true)]
