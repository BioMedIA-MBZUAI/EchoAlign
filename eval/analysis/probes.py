import warnings

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

C_GRID = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
DEFAULT_C, DEFAULT_ALPHA = 1.0, 1.0
MIN_VAL = 1


def _fit_logistic(model, x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x, y)
        retried = False
        if int(model.n_iter_.max()) >= model.max_iter:
            model.set_params(max_iter=10000).fit(x, y)
            retried = True
    return int(model.n_iter_.max()) < model.max_iter, retried


def _better(score, best):
    return score is not None and np.isfinite(score) and (best is None or not np.isfinite(best) or score > best)


def select_logistic(tr, y_train, score, usable, name):
    best_c, best_score, model, curve = None, None, None, []
    for c in (C_GRID if usable else [DEFAULT_C]):
        if model is None:
            model = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", tol=1e-4, warm_start=True)
        else:
            model.set_params(C=c, max_iter=2000)
        converged, retried = _fit_logistic(model, tr, y_train)
        s = float(score(model)) if usable else None
        curve.append({"C": c, name: s, "converged": converged, "retried_10000": retried})
        if converged and (best_c is None or (usable and _better(s, best_score))):
            best_c, best_score = c, s
    if best_c is None:
        raise RuntimeError("no converged logistic regression fit")
    return best_c, best_score, curve


def fit_logistic(c, x, y):
    head = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", tol=1e-4, warm_start=True)
    converged, retried = _fit_logistic(head, x, y)
    if not converged:
        raise RuntimeError("final logistic regression fit did not converge")
    return head, retried


def classification_probe(x_train, y_train, x_val, y_val, x_test):
    y_train, y_val = np.asarray(y_train, int), np.asarray(y_val, int)
    sc = StandardScaler().fit(np.asarray(x_train, np.float64))
    tr, va = sc.transform(np.asarray(x_train, np.float64)), sc.transform(np.asarray(x_val, np.float64))
    usable = len(y_val) >= MIN_VAL and len(np.unique(y_val)) == 2
    best_c, best_score, curve = select_logistic(
        tr, y_train, lambda m: roc_auc_score(y_val, m.predict_proba(va)[:, 1]), usable, "validation_auroc")
    x_fit = np.concatenate([x_train, x_val]).astype(np.float64)
    y_fit = np.concatenate([y_train, y_val])
    sc = StandardScaler().fit(x_fit)
    head, retried = fit_logistic(best_c, sc.transform(x_fit), y_fit)
    pred = expit(head.decision_function(sc.transform(np.asarray(x_test, np.float64))))
    return pred, dict(C=best_c, C_source="validation" if usable else "default", validation_auroc=best_score,
                      final_retry_10000=retried, curve=curve)


def regression_probe(x_train, y_train, x_val, y_val, x_test):
    y_train, y_val = np.asarray(y_train, np.float64), np.asarray(y_val, np.float64)
    sc = StandardScaler().fit(np.asarray(x_train, np.float64))
    tr, va = sc.transform(np.asarray(x_train, np.float64)), sc.transform(np.asarray(x_val, np.float64))
    usable = len(y_val) >= MIN_VAL
    best, best_rmse, curve = None, None, []
    for a in (ALPHA_GRID if usable else [DEFAULT_ALPHA]):
        fit = Ridge(alpha=a, solver="cholesky").fit(tr, y_train)
        rmse = float(mean_squared_error(y_val, fit.predict(va)) ** 0.5) if usable else None
        curve.append(dict(alpha=a, validation_rmse=rmse))
        if best is None or (usable and _better(-rmse, None if best_rmse is None else -best_rmse)):
            best, best_rmse = a, rmse
    x_fit = np.concatenate([x_train, x_val]).astype(np.float64)
    sc = StandardScaler().fit(x_fit)
    head = Ridge(alpha=best, solver="cholesky").fit(sc.transform(x_fit), np.concatenate([y_train, y_val]))
    pred = head.predict(sc.transform(np.asarray(x_test, np.float64)))
    return pred, dict(alpha=best, alpha_source="validation" if usable else "default", validation_rmse=best_rmse,
                      curve=curve)
