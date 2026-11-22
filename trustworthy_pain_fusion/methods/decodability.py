"""Temperature-calibrated usable information and permutation tests."""
import os
from functools import partial
import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from trustworthy_pain_fusion.methods.splits import grouped_kfold
EPS = 1e-12
N_PERM = 100
N_JOBS = int(os.environ.get('DECOD_NJOBS', '1'))

def _entropy_bits(y):
    _, c = np.unique(y, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())

def _temper(proba, T):
    p = np.clip(proba, EPS, 1) ** (1.0 / T)
    return p / p.sum(1, keepdims=True)

def _ce_bits(proba, y, classes):
    idx = np.array([classes.index(v) for v in y])
    return float(-np.mean(np.log2(np.clip(proba[np.arange(len(idx)), idx], EPS, 1))))

def _fit_temperature(proba, y, classes):
    res = minimize_scalar(lambda T: _ce_bits(_temper(proba, T), y, classes), bounds=(0.25, 10.0), method='bounded')
    return float(res.x)

def _lgbm(n_jobs=4):
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight='balanced', n_jobs=n_jobs, verbose=-1)

def inner_split(tr, subj, rng, frac=0.8):
    tr = np.asarray(tr)
    s = np.asarray(subj)[tr]
    uniq = np.unique(s)
    if len(uniq) < 2:
        return (tr, tr)
    order = rng.permutation(uniq)
    n_fit = min(len(uniq) - 1, max(1, int(round(frac * len(uniq)))))
    fit_mask = np.isin(s, order[:n_fit])
    return (tr[fit_mask], tr[~fit_mask])

def permute_within(y, subj, rng):
    y = np.asarray(y)
    subj = np.asarray(subj)
    yp = y.copy()
    for s in np.unique(subj):
        idx = np.flatnonzero(subj == s)
        yp[idx] = y[idx][rng.permutation(len(idx))]
    return yp

def null_labels(y, subj, n_perm, seed=0, scheme='within'):
    rng = np.random.default_rng(seed)
    if scheme == 'within':
        return [permute_within(y, subj, rng) for _ in range(n_perm)]
    raise ValueError('only within-subject permutation is supported')

def _cv_eval(X, yt, subj, k, seed, model_factory, n_classes):
    rng = np.random.default_rng(seed)
    ces, accs, Y, preds, P = ([], [], [], [], [])
    classes = None
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = inner_split(tr, subj, rng)
        if len(np.unique(yt[fit_i])) < n_classes or len(cal_i) == 0:
            fit_i = cal_i = np.asarray(tr)
        clf = model_factory().fit(X[fit_i], yt[fit_i])
        classes = list(clf.classes_)
        T = _fit_temperature(clf.predict_proba(X[cal_i]), yt[cal_i], classes)
        proba_te = _temper(clf.predict_proba(X[te]), T)
        ces.append(_ce_bits(proba_te, yt[te], classes))
        pred = clf.predict(X[te])
        accs.append(float((pred == yt[te]).mean()))
        Y.append(yt[te])
        preds.append(pred)
        P.append(proba_te)
    return (float(np.mean(ces)), np.asarray(ces), float(np.mean(accs)), np.concatenate(Y), np.concatenate(preds), np.concatenate(P), classes)

def _null_ce(X, yt, subj, k, seed, model_factory, n_classes):
    return _cv_eval(X, yt, subj, k, seed, model_factory, n_classes)[0]

def decodability_profile(X, y, subj, k=5, seed=0, n_perm=N_PERM, model_factory=None, perm_scheme='within', n_jobs=None):
    y = np.asarray(y).astype(int)
    subj = np.asarray(subj)
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    parallel = n_jobs > 1 and n_perm > 1
    if model_factory is None:
        model_factory = partial(_lgbm, n_jobs=1 if parallel else 4)
    n_classes = int(len(np.unique(y)))
    H = _entropy_bits(y)
    ce, ces_folds, acc, Y, pred, P, classes = _cv_eval(X, y, subj, k, seed, model_factory, n_classes)
    perms = null_labels(y, subj, n_perm, seed=seed, scheme=perm_scheme)
    if parallel:
        null_ce = Parallel(n_jobs=n_jobs)((delayed(_null_ce)(X, yp, subj, k, seed, model_factory, n_classes) for yp in perms))
    else:
        null_ce = [_null_ce(X, yp, subj, k, seed, model_factory, n_classes) for yp in perms]
    null_iv = H - np.asarray(null_ce, float)
    iv_unclamped = H - ce
    iv_folds = H - ces_folds
    out = {'n_classes': n_classes, 'H_Y_bits': round(H, 3), 'usable_info_bits': round(max(0.0, iv_unclamped), 3), 'usable_info_std': round(float(np.std(iv_folds)), 3), 'usable_info_unclamped_bits': round(iv_unclamped, 3), 'usable_info_frac_of_H': round(max(0.0, iv_unclamped) / H, 3) if H else 0.0, 'cross_entropy_bits': round(ce, 3), 'cv_accuracy': round(acc, 3), 'bayes_error_proxy': round(1 - acc, 3), 'cv_balanced_accuracy': round(float(balanced_accuracy_score(Y, pred)), 3), 'majority_rate': round(float(np.bincount(y).max() / len(y)), 3), 'n_perm': int(n_perm), 'perm_scheme': perm_scheme, 'calibration_split': 'subject-grouped (about 80% of training subjects fit, 20% calibrate)'}
    if n_perm:
        out.update({'perm_null_usable_info_bits': round(float(np.mean(np.maximum(null_iv, 0.0))), 3), 'perm_null_unclamped_mean_bits': round(float(null_iv.mean()), 3), 'perm_null_unclamped_p95_bits': round(float(np.percentile(null_iv, 95)), 3), 'usable_info_p_value': round(float((np.sum(null_iv >= iv_unclamped) + 1) / (n_perm + 1)), 4)})
    if n_classes == 2 and classes is not None and (1 in classes):
        out['cv_auc'] = round(float(roc_auc_score(Y, P[:, classes.index(1)])), 3)
    return out

def ai4pain_targets(y):
    y = np.asarray(y)
    pain = y != 0
    return {'presence (NoPain vs Pain)': ((y != 0).astype(int), np.ones(len(y), bool)), 'location (Arm vs Hand)': ((y[pain] == 2).astype(int), pain)}



class UsableInformationEstimator:
    """Temperature-calibrated usable information with a within-subject permutation null."""

    def __init__(self, k=5, seed=0, n_perm=N_PERM, model_factory=None, perm_scheme='within', n_jobs=None):
        self.k, self.seed, self.n_perm = k, seed, n_perm
        self.model_factory, self.perm_scheme, self.n_jobs = model_factory, perm_scheme, n_jobs

    def profile(self, X, y, subj):
        return decodability_profile(X, y, subj, k=self.k, seed=self.seed, n_perm=self.n_perm,
                                    model_factory=self.model_factory, perm_scheme=self.perm_scheme,
                                    n_jobs=self.n_jobs)
