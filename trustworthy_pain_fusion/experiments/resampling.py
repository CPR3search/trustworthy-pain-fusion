"""Subject bootstrap intervals and offline confidence ranking."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.methods.splits import grouped_kfold
from trustworthy_pain_fusion.methods.decodability import _entropy_bits, _ce_bits, _temper, _fit_temperature, _lgbm
from trustworthy_pain_fusion.methods.fusion import load_bundle, _cal_proba, _cal_proba_pair, _ds_combine, _inner, _reliability
ROOT = _PROJECT_ROOT
RESULTS = ROOT / 'results'
SEED = 0

def _oof_calibrated(X, y, subj, k=5, seed=SEED):
    y = np.asarray(y).astype(int)
    n = len(y)
    P = np.full((n, 2), 0.5)
    rng = np.random.default_rng(seed)
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, y, rng, 2)
        P[te] = _cal_proba(X, y, fit_i, cal_i, te)
    return P

def _boot_subject_idx(subj, rng):
    uniq = np.unique(subj)
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([np.where(subj == s)[0] for s in pick])

def bootstrap_ci(X, y, subj, B=400, seed=SEED):
    y = np.asarray(y).astype(int)
    subj = np.asarray(subj)
    P = _oof_calibrated(X, y, subj, seed=seed)
    classes = [0, 1]
    pred = P.argmax(1)
    auc_score = np.full(len(y), 0.5)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, y, subj):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        rf = RandomForestClassifier(n_estimators=200, n_jobs=4, random_state=seed)
        rf.fit(sc.transform(X[tr]), y[tr])
        auc_score[te] = rf.predict_proba(sc.transform(X[te]))[:, 1]

    def stats(idx):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return None
        iv = _entropy_bits(yy) - _ce_bits(P[idx], yy, classes)
        acc = float((pred[idx] == yy).mean())
        auc = float(roc_auc_score(yy, auc_score[idx]))
        return (iv, acc, auc)
    point = stats(np.arange(len(y)))
    rng = np.random.default_rng(seed)
    boots = [stats(_boot_subject_idx(subj, rng)) for _ in range(B)]
    arr = np.array([b for b in boots if b is not None])
    ci = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {'n_subjects': int(len(np.unique(subj))), 'n_boot': int(len(arr)), 'resampling': 'subjects resampled over fixed out-of-fold predictions (no refit)', 'iv_unclamped_bits': round(point[0], 3), 'iv_unclamped_ci95': ci(arr[:, 0]), 'iv_bits': round(max(0.0, point[0]), 3), 'iv_ci95': ci(np.maximum(arr[:, 0], 0.0)), 'accuracy': round(point[1], 3), 'acc_ci95': ci(arr[:, 1]), 'c2st_auc': round(point[2], 3), 'auc_ci95': ci(arr[:, 2])}

def selective(X, y, subj, coverages=(1.0, 0.8, 0.6, 0.4, 0.2), seed=SEED):
    y = np.asarray(y).astype(int)
    P = _oof_calibrated(X, y, subj, seed=seed)
    conf = P.max(1)
    pred = P.argmax(1)
    correct = (pred == y).astype(float)
    order = np.argsort(-conf)
    out = []
    for c in coverages:
        m = max(1, int(round(c * len(y))))
        kept = order[:m]
        bacc = round(float(balanced_accuracy_score(y[kept], pred[kept])), 3) if len(np.unique(y[kept])) > 1 else None
        out.append({'coverage': c, 'accuracy': round(float(correct[kept].mean()), 3), 'balanced_accuracy': bacc, 'n_kept': int(m)})
    return out

def missing_impute(dataset, target, drop_mod, k=5, seed=SEED):
    b = load_bundle(dataset)
    yt, mask = b['targets'][target]
    yt = np.asarray(yt).astype(int)
    X = b['X'][mask]
    subj = b['si'][mask]
    mods = b['mods']
    fused = sorted({i for _, ci in mods for i in ci})
    drop_cols = dict(mods)[drop_mod]
    drop_local = [j for j, c in enumerate(fused) if c in set(drop_cols)]
    keep_mods = [(mn, ci) for mn, ci in mods if mn != drop_mod]
    rng = np.random.default_rng(seed)
    preds = {'early_mean': [], 'early_knn': [], 'evidential_votedrop': [], 'full_early': []}
    Y = []
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, yt, rng, 2)
        Xf = X[:, fused]
        Y.append(yt[te])
        clf = _lgbm().fit(Xf[fit_i], yt[fit_i])
        preds['full_early'].append(clf.predict(Xf[te]))
        mean = Xf[fit_i].mean(0)
        Xm = Xf[te].copy()
        Xm[:, drop_local] = mean[drop_local]
        preds['early_mean'].append(clf.predict(Xm))
        imp = KNNImputer(n_neighbors=10).fit(Xf[fit_i])
        Xk = Xf[te].copy().astype(float)
        Xk[:, drop_local] = np.nan
        preds['early_knn'].append(clf.predict(imp.transform(Xk)))
        probs, rel = ([], [])
        for mn, ci in keep_mods:
            p_cal, p_te = _cal_proba_pair(X[:, ci], yt, fit_i, cal_i, te)
            probs.append(p_te)
            rel.append(_reliability(p_cal, yt[cal_i]))
        preds['evidential_votedrop'].append(_ds_combine(probs, rel)[0].argmax(1))
    yy = np.concatenate(Y)
    out = {}
    for name, parts in preds.items():
        p = np.concatenate(parts)
        out[name] = round(float((p == yy).mean()), 3)
        out[f'{name}_bacc'] = round(float(balanced_accuracy_score(yy, p)), 3)
    return out

def main():
    out = {'protocol': 'subject-grouped 5-fold with a subject-grouped calibration split; subject-level bootstrap (B=400) over fixed out-of-fold predictions; risk-coverage from the calibrated early-fusion confidence; missing = mean vs kNN imputation vs evidential vote-drop'}
    b26 = load_bundle('ai4pain2026')
    ci = {}
    for tgt in ('presence', 'location'):
        yt, mask = b26['targets'][tgt]
        ci[tgt] = bootstrap_ci(b26['X'][mask], np.asarray(yt), b26['si'][mask])
        print(f"[M2b] 2026 {tgt}: I_V(unclamped)={ci[tgt]['iv_unclamped_bits']} {ci[tgt]['iv_unclamped_ci95']} acc={ci[tgt]['accuracy']} {ci[tgt]['acc_ci95']} AUC={ci[tgt]['c2st_auc']} {ci[tgt]['auc_ci95']}", flush=True)
    out['bootstrap_ci'] = {'ai4pain2026': ci}
    sel = {}
    for ds, tgt, key in [('ai4pain2026', 'presence', 'ai4pain2026:presence'), ('ai4pain2026', 'location', 'ai4pain2026:location'), ('ai4pain2025', 'intensity', 'ai4pain2025:intensity'), ('biovid', 'presence', 'biovid:presence')]:
        bb = load_bundle(ds)
        yt, mask = bb['targets'][tgt]
        sel[key] = selective(bb['X'][mask], np.asarray(yt), bb['si'][mask])
        print(f'[M3] {key}: ' + ' '.join((f"{s['coverage']}:{s['balanced_accuracy']}" for s in sel[key])), flush=True)
    out['selective'] = sel
    mi = {}
    for ds, tgt, mod in [('ai4pain2026', 'presence', 'Bvp'), ('ai4pain2026', 'presence', 'Eda'), ('ai4pain2025', 'intensity', 'Eda'), ('biovid', 'presence', 'gsr')]:
        r = missing_impute(ds, tgt, mod)
        mi[f'{ds}:{tgt}:-{mod}'] = r
        print(f"[M3b] {ds}:{tgt} -{mod}: mean={r['early_mean_bacc']} knn={r['early_knn_bacc']} votedrop={r['evidential_votedrop_bacc']} (full={r['full_early_bacc']}) [balanced]", flush=True)
    out['missing_impute'] = mi
    (RESULTS / 'resampling.json').write_text(json.dumps(out, indent=2))
    print('wrote results/resampling.json', flush=True)
if __name__ == '__main__':
    main()
