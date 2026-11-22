"""Repeated partitions, sensor loss, degradation, normalization and frozen calibration (runs 1, 3-6)."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import argparse
import json
import time
from functools import partial
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.methods.splits import grouped_kfold
from trustworthy_pain_fusion.methods.decodability import _entropy_bits, _temper, _ce_bits, _fit_temperature, _lgbm, inner_split
from trustworthy_pain_fusion.methods.fusion import COMBINERS, _inner, _cal_proba_pair, _combine, _ds_combine, _reliability, _prob_metrics, _paired_bacc, _iv_on, _iv_summary, _fusion_methods_one, _mod_cols, load_bundle
ROOT = _PROJECT_ROOT
RESULTS = ROOT / 'results'
DROPOUT_P = 0.25
DROPOUT_COPIES = 3
KNN_K = 10
BASELINE_NORM_IDS = {'ai4pain2026': ('_Baseline_1', '_Baseline_2'), 'ai4pain2025': ('#w0', '#w1')}
LOAD_BEARING = {('ai4pain2026', 'presence'): 'Bvp', ('ai4pain2025', 'presence'): 'Bvp', ('ai4pain2025', 'intensity'): 'Eda', ('biovid', 'presence'): 'gsr'}

def _is_norm_window(dataset, seg_id):
    if dataset == 'ai4pain2026':
        return seg_id.endswith(BASELINE_NORM_IDS[dataset])
    if dataset == 'ai4pain2025':
        return '_Baseline_' in seg_id and seg_id.endswith(BASELINE_NORM_IDS[dataset])
    return False

def normalize_from_windows(segments, subjects, signals, use):
    subjects = np.asarray(subjects)
    use = np.asarray(use, bool)
    out = [dict(seg) for seg in segments]
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        ref = idx[use[idx]]
        if len(ref) == 0:
            raise ValueError(f'subject {s}: no normalization windows')
        for sig in signals:
            vals = np.concatenate([segments[i][sig] for i in ref])
            mu, sd = (vals.mean(), vals.std() + 1e-08)
            for i in idx:
                out[i][sig] = (segments[i][sig] - mu) / sd
    return out

def load_bundle_norm(dataset, norm):
    if dataset == 'ai4pain2026':
        from trustworthy_pain_fusion.data import ai4pain2026 as mod
        segs, subj, lab, ids = mod.load_segments('data', 'train', mod.SIGNALS)
        signals = mod.SIGNALS
        names = mod.feature_names(mod.SIGNALS)
        build = partial(mod.build_features, signals=mod.SIGNALS)
        mod_names = ['Bvp', 'Eda', 'Resp', 'SpO2']
    elif dataset == 'ai4pain2025':
        from trustworthy_pain_fusion.data import ai4pain2025 as mod
        segs, subj, lab, ids = mod.load_segments(split='train', nopain_mode='baseline_seg')
        signals = mod.SIGNALS
        names = mod.feature_names()
        build = mod.build_features
        mod_names = ['Bvp', 'Eda', 'Resp', 'SpO2']
    elif dataset == 'biovid':
        from trustworthy_pain_fusion.data import biovid as mod
        segs, subj, lab, ids = mod.load_segments()
        signals = mod.CHANNELS
        names = mod.feature_names()
        build = mod.build_features
        mod_names = ['gsr', 'ecg', 'emg_trapezius']
    else:
        raise ValueError(dataset)
    flag = np.array([_is_norm_window(dataset, i) for i in ids])
    if dataset != 'biovid':
        per_subj = {s: flag[np.asarray(subj) == s].sum() for s in np.unique(subj)}
        assert all((v == 2 for v in per_subj.values())), f'{dataset}: expected 2 norm windows/subject'
    if norm == 'subject_all':
        segs = mod.normalize_per_subject(segs, subj, signals)
    elif norm == 'baseline_only':
        if dataset != 'biovid':
            segs = normalize_from_windows(segs, subj, signals, flag)
    else:
        raise ValueError(norm)
    keep = ~flag
    segs = [s for s, k in zip(segs, keep) if k]
    subj = np.asarray(subj)[keep]
    y = np.asarray(lab, int)[keep]
    X = np.asarray(build(segs), float)
    if dataset == 'biovid':
        m = (y == 0) | (y == 4)
        ones = np.ones(len(y), bool)
        targets = {'presence': ((y[m] == 4).astype(int), m), 'intensity': (y.astype(int), ones)}
    else:
        pain = y != 0
        ones = np.ones(len(y), bool)
        second = 'location' if dataset == 'ai4pain2026' else 'intensity'
        targets = {'presence': ((y != 0).astype(int), ones), second: ((y[pain] == 2).astype(int), pain)}
    mods = [(mn, _mod_cols(names, mn)) for mn in mod_names]
    mods = [(mn, ci) for mn, ci in mods if ci]
    _, si = np.unique(subj, return_inverse=True)
    return {'X': X, 'names': names, 'mods': mods, 'si': si, 'targets': targets, 'n_dropped_norm_windows': int(flag.sum())}

def brier_skill_reliability(p_cal, y_cal, prior1):
    y_cal = np.asarray(y_cal, int)
    idx = np.arange(len(y_cal))

    def brier(P):
        return float(np.mean((P[idx, y_cal] - 1) ** 2 + P[idx, 1 - y_cal] ** 2))
    prior = np.tile([1 - prior1, prior1], (len(y_cal), 1))
    b_prior = brier(prior)
    if b_prior <= 0:
        return 0.0
    return float(np.clip(1 - brier(p_cal) / b_prior, 0.0, 1.0))

def dropout_masks(n, n_mods, p, rng):
    avail = rng.random((n, n_mods)) >= p
    bad = ~avail.any(1)
    while bad.any():
        avail[bad] = rng.random((bad.sum(), n_mods)) >= p
        bad = ~avail.any(1)
    return avail

def _apply_avail(Xf, local_blocks, avail, fill):
    Xo = Xf.copy()
    for j, loc in enumerate(local_blocks):
        miss = ~avail[:, j]
        if miss.any():
            Xo[np.ix_(miss, loc)] = fill[loc]
    return np.hstack([Xo, avail.astype(float)])

def missing_fusion_rev(X, y, subj, mods, k=5, seed=0, dominant=None, extra=True):
    fused_cols = sorted({i for _, ci in mods for i in ci})
    modlist = [mn for mn, _ in mods]
    local_blocks = [[j for j, c in enumerate(fused_cols) if c in set(ci)] for _, ci in mods]
    rng = np.random.default_rng(seed)
    drng = np.random.default_rng(10000 + seed)
    scen = ['none'] + [f'drop:{m}' for m in modlist]
    combiners = list(COMBINERS) + ([('evidential', 'evidential_brier')] if extra else [])
    base_ops = [n for _, n in combiners]
    ops = {s: (['early'] if s == 'none' else ['early_impute']) + base_ops for s in scen}
    if extra:
        ops['none'] += ['early_dropout']
        for s in scen[1:]:
            ops[s] += ['early_dropout', 'early_knn']
    parts = {s: {op: [] for op in ops[s]} for s in scen}
    Ys, Ss, rels = ([], [], [])
    subj = np.asarray(subj)
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, y, rng, 2)
        Ys.append(y[te])
        Ss.append(subj[te])
        P, R, RB = ({}, {}, {})
        prior1 = float(np.mean(y[fit_i]))
        for mn, ci in mods:
            pc, pt = _cal_proba_pair(X[:, ci], y, fit_i, cal_i, te)
            P[mn] = pt
            R[mn] = _reliability(pc, y[cal_i])
            if extra:
                RB[mn] = brier_skill_reliability(pc, y[cal_i], prior1)
        rels.append({'ba': R, 'brier': RB})
        Xf = X[:, fused_cols]
        clf = _lgbm().fit(Xf[fit_i], y[fit_i])
        classes = list(clf.classes_)
        T = _fit_temperature(clf.predict_proba(Xf[cal_i]), y[cal_i], classes)
        parts['none']['early'].append(_temper(clf.predict_proba(Xf[te]), T))

        def comb(keep, how, name):
            w = RB if name == 'evidential_brier' else R
            return _combine([P[m] for m in keep], [w[m] for m in keep], how)
        for how, name in combiners:
            parts['none'][name].append(comb(modlist, how, name))
        mean = Xf[fit_i].mean(0)
        if extra:
            n_fit = len(fit_i)
            M = len(mods)
            av = [np.ones((n_fit, M), bool)] + [dropout_masks(n_fit, M, DROPOUT_P, drng) for _ in range(DROPOUT_COPIES)]
            Xtr = np.vstack([_apply_avail(Xf[fit_i], local_blocks, a, mean) for a in av])
            ytr = np.tile(y[fit_i], DROPOUT_COPIES + 1)
            dclf = _lgbm().fit(Xtr, ytr)
            full_av_cal = np.ones((len(cal_i), M), bool)
            Td = _fit_temperature(dclf.predict_proba(_apply_avail(Xf[cal_i], local_blocks, full_av_cal, mean)), y[cal_i], list(dclf.classes_))
            full_av_te = np.ones((len(te), M), bool)
            parts['none']['early_dropout'].append(_temper(dclf.predict_proba(_apply_avail(Xf[te], local_blocks, full_av_te, mean)), Td))
            knn = KNNImputer(n_neighbors=KNN_K).fit(Xf[fit_i])
        for j, (drop, ci) in enumerate(mods):
            s = f'drop:{drop}'
            local = local_blocks[j]
            Xte = Xf[te].copy()
            Xte[:, local] = mean[local]
            parts[s]['early_impute'].append(_temper(clf.predict_proba(Xte), T))
            keep = [m for m in modlist if m != drop]
            for how, name in combiners:
                parts[s][name].append(comb(keep, how, name))
            if extra:
                av_te = np.ones((len(te), len(mods)), bool)
                av_te[:, j] = False
                parts[s]['early_dropout'].append(_temper(dclf.predict_proba(_apply_avail(Xf[te], local_blocks, av_te, mean)), Td))
                Xk = Xf[te].copy().astype(float)
                Xk[:, local] = np.nan
                parts[s]['early_knn'].append(_temper(clf.predict_proba(knn.transform(Xk)), T))
    y_all = np.concatenate(Ys)
    s_all = np.concatenate(Ss)
    res, pred = ({}, {})
    for s in scen:
        res[s] = {}
        pred[s] = {}
        for op in ops[s]:
            Pc = np.concatenate(parts[s][op])
            m = _prob_metrics(Pc, y_all)
            res[s][op] = {'acc': m['acc'], 'bacc': m['bacc'], 'ece': m['ece'], 'brier': m['brier']}
            pred[s][op] = Pc.argmax(1)
    pairs = [('evidential', 'early_impute'), ('late_mean', 'early_impute'), ('late_mean', 'evidential')]
    if extra:
        pairs += [('evidential', 'early_dropout'), ('early_dropout', 'early_impute'), ('evidential', 'early_knn'), ('evidential_brier', 'evidential'), ('evidential_brier', 'early_impute')]
    res['paired_tests_bacc'] = {s: {f'{a}_minus_{b}': _paired_bacc(pred[s][a], pred[s][b], y_all, s_all) for a, b in pairs} for s in scen if s != 'none'}
    if extra:
        res['paired_tests_bacc']['none'] = {f'{a}_minus_{b}': _paired_bacc(pred['none'][a], pred['none'][b], y_all, s_all) for a, b in [('early_dropout', 'early'), ('evidential', 'early'), ('evidential_brier', 'evidential')]}
        res['reliability_mean'] = {kind: {m: round(float(np.mean([r[kind][m] for r in rels])), 3) for m in modlist} for kind in ('ba', 'brier')}
    res['dominant_modality'] = dominant
    return res

def run1_one(dataset, seed):
    t0 = time.time()
    b = load_bundle(dataset)
    out = {}
    fused_cols = sorted({i for _, ci in b['mods'] for i in ci})
    for tname, (yt, mask) in b['targets'].items():
        yt = np.asarray(yt).astype(int)
        subj = b['si'][mask]
        Xt = b['X'][mask]
        per_mod = {mn: _iv_summary(_iv_on(Xt[:, ci], yt, subj, seed=seed)) for mn, ci in b['mods']}
        fused = _iv_summary(_iv_on(Xt[:, fused_cols], yt, subj, seed=seed))
        allf = _iv_summary(_iv_on(Xt, yt, subj, seed=seed))
        best = max(per_mod, key=lambda m: per_mod[m]['iv_unclamped'])
        r = {'per_modality': per_mod, 'fused': fused, 'all_features': allf, 'best_single_modality': best, 'fusion_gain_unclamped_bits': round(fused['iv_unclamped'] - per_mod[best]['iv_unclamped'], 3)}
        if len(np.unique(yt)) == 2:
            r['operators_complete'] = _fusion_methods_one(Xt, yt, subj, b['mods'], seed=seed)
            r['missing'] = missing_fusion_rev(Xt, yt, subj, b['mods'], seed=seed, dominant=LOAD_BEARING.get((dataset, tname)))
        out[tname] = r
    print(f'[run1] {dataset} seed={seed} done in {time.time() - t0:.0f}s', flush=True)
    return (dataset, seed, out)
PREDICTORS = {'logreg': lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight='balanced')), 'random_forest': lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced', n_jobs=4, random_state=0), 'lightgbm': lambda: _lgbm(n_jobs=4)}

def iv_raw_and_tempered(X, y, subj, factory, k=5, seed=0):
    y = np.asarray(y).astype(int)
    n_classes = len(np.unique(y))
    H = _entropy_bits(y)
    rng = np.random.default_rng(seed)
    ce_raw, ce_t, Y, pred, P = ([], [], [], [], [])
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = inner_split(tr, subj, rng)
        if len(np.unique(y[fit_i])) < n_classes or len(cal_i) == 0:
            fit_i = cal_i = np.asarray(tr)
        clf = factory().fit(X[fit_i], y[fit_i])
        classes = list(clf.classes_)
        T = _fit_temperature(clf.predict_proba(X[cal_i]), y[cal_i], classes)
        p_raw = clf.predict_proba(X[te])
        p_t = _temper(p_raw, T)
        ce_raw.append(_ce_bits(p_raw, y[te], classes))
        ce_t.append(_ce_bits(p_t, y[te], classes))
        Y.append(y[te])
        pred.append(clf.predict(X[te]))
        P.append(p_t)
    Y = np.concatenate(Y)
    pred = np.concatenate(pred)
    P = np.concatenate(P)
    out = {'H_Y_bits': round(H, 3), 'uncalibrated': {'ce_bits': round(float(np.mean(ce_raw)), 3), 'iv_unclamped': round(H - float(np.mean(ce_raw)), 3)}, 'temperature': {'ce_bits': round(float(np.mean(ce_t)), 3), 'iv_unclamped': round(H - float(np.mean(ce_t)), 3)}, 'bacc': round(float(balanced_accuracy_score(Y, pred)), 3)}
    if n_classes == 2:
        out['auc'] = round(float(roc_auc_score(Y, P[:, 1])), 3)
    return out

def run3_one(dataset, tname, pname):
    b = load_bundle(dataset)
    yt, mask = b['targets'][tname]
    r = iv_raw_and_tempered(b['X'][mask], yt, b['si'][mask], PREDICTORS[pname])
    print(f"[run3] {dataset} {tname} {pname}: I_V raw={r['uncalibrated']['iv_unclamped']} temp={r['temperature']['iv_unclamped']} bacc={r['bacc']}", flush=True)
    return (dataset, tname, pname, r)

def run5_one(dataset, norm):
    t0 = time.time()
    b = load_bundle_norm(dataset, norm)
    out = {'n_dropped_norm_windows': b['n_dropped_norm_windows']}
    for tname, (yt, mask) in b['targets'].items():
        yt = np.asarray(yt).astype(int)
        subj = b['si'][mask]
        Xt = b['X'][mask]
        r = {'all_features': _iv_summary(_iv_on(Xt, yt, subj, seed=0)), 'n_windows': int(len(yt)), 'class_counts': np.bincount(yt).tolist()}
        if len(np.unique(yt)) == 2:
            r['operators_complete'] = _fusion_methods_one(Xt, yt, subj, b['mods'], seed=0)
            r['missing'] = missing_fusion_rev(Xt, yt, subj, b['mods'], seed=0, dominant=LOAD_BEARING.get((dataset, tname)))
        out[tname] = r
    print(f'[run5] {dataset} norm={norm} done in {time.time() - t0:.0f}s', flush=True)
    return (dataset, norm, out)
CORRUPTIONS = [('noise_snr10', 'noise', 10.0), ('noise_snr5', 'noise', 5.0), ('noise_snr0', 'noise', 0.0), ('gap50', 'gap', 0.5)]
CORRUPTION_SEED = 20261117
LB_CHANNEL = {'Bvp': 'Bvp', 'Eda': 'Eda', 'gsr': 'gsr'}

def corrupt_signal(x, kind, level, rng):
    x = np.asarray(x, float)
    if kind == 'noise':
        power = float(np.var(x))
        sigma = np.sqrt(power / 10.0 ** (level / 10.0))
        return x + rng.normal(0.0, sigma, len(x))
    if kind == 'gap':
        n = len(x)
        g = int(round(level * n))
        start = int(rng.integers(0, n - g + 1))
        y = x.copy()
        y[start:start + g] = 0.0
        return y
    raise ValueError(kind)

def load_clean(dataset):
    if dataset == 'ai4pain2026':
        from trustworthy_pain_fusion.data import ai4pain2026 as mod
        segs, subj, lab, _ = mod.load_segments('data', 'train', mod.SIGNALS)
        segs = mod.normalize_per_subject(segs, subj, mod.SIGNALS)
        build = partial(mod.build_features, signals=mod.SIGNALS)
    elif dataset == 'ai4pain2025':
        from trustworthy_pain_fusion.data import ai4pain2025 as mod
        segs, subj, lab, _ = mod.load_segments(split='train', nopain_mode='baseline_seg')
        segs = mod.normalize_per_subject(segs, subj)
        build = mod.build_features
    elif dataset == 'biovid':
        from trustworthy_pain_fusion.data import biovid as mod
        segs, subj, lab, _ = mod.load_segments()
        segs = mod.normalize_per_subject(segs, subj)
        build = mod.build_features
    else:
        raise ValueError(dataset)
    return (segs, build)

def corrupted_features(segs, build, idx, channel):
    out = {}
    for ci, (name, kind, level) in enumerate(CORRUPTIONS):
        cs = []
        for i in idx:
            rng = np.random.default_rng([CORRUPTION_SEED, ci, int(i)])
            seg = dict(segs[i])
            seg[channel] = corrupt_signal(segs[i][channel], kind, level, rng)
            cs.append(seg)
        out[name] = np.asarray(build(cs), float)
    return out

def _fit_cal(X, y, fit_i, cal_i):
    clf = _lgbm().fit(X[fit_i], y[fit_i])
    classes = list(clf.classes_)
    pc = clf.predict_proba(X[cal_i])
    T = _fit_temperature(pc, y[cal_i], classes)
    return (clf, T, _temper(pc, T))

def corruption_eval(X, Xc, y, subj, mods, k=5, seed=0):
    fused_cols = sorted({i for _, ci in mods for i in ci})
    modlist = [mn for mn, _ in mods]
    local_blocks = [[j for j, c in enumerate(fused_cols) if c in set(ci)] for _, ci in mods]
    rng = np.random.default_rng(seed)
    drng = np.random.default_rng(10000 + seed)
    conds = ['clean'] + [c[0] for c in CORRUPTIONS]
    ops = ['early', 'early_dropout', 'late_mean', 'evidential', 'evidential_brier']
    parts = {c: {op: [] for op in ops} for c in conds}
    Ys, Ss = ([], [])
    subj = np.asarray(subj)
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, y, rng, 2)
        Ys.append(y[te])
        Ss.append(subj[te])
        prior1 = float(np.mean(y[fit_i]))
        M = {}
        for mn, ci in mods:
            clf, T, pc = _fit_cal(X[:, ci], y, fit_i, cal_i)
            M[mn] = (clf, T, ci, _reliability(pc, y[cal_i]), brier_skill_reliability(pc, y[cal_i], prior1))
        Xf = X[:, fused_cols]
        eclf, eT, _ = _fit_cal(Xf, y, fit_i, cal_i)
        mean = Xf[fit_i].mean(0)
        nM = len(mods)
        av = [np.ones((len(fit_i), nM), bool)] + [dropout_masks(len(fit_i), nM, DROPOUT_P, drng) for _ in range(DROPOUT_COPIES)]
        dclf = _lgbm().fit(np.vstack([_apply_avail(Xf[fit_i], local_blocks, a, mean) for a in av]), np.tile(y[fit_i], DROPOUT_COPIES + 1))
        Td = _fit_temperature(dclf.predict_proba(_apply_avail(Xf[cal_i], local_blocks, np.ones((len(cal_i), nM), bool), mean)), y[cal_i], list(dclf.classes_))
        for c in conds:
            Xe = X[te] if c == 'clean' else Xc[c][te]
            Xfe = Xe[:, fused_cols]
            parts[c]['early'].append(_temper(eclf.predict_proba(Xfe), eT))
            parts[c]['early_dropout'].append(_temper(dclf.predict_proba(_apply_avail(Xfe, local_blocks, np.ones((len(te), nM), bool), mean)), Td))
            P = {mn: _temper(M[mn][0].predict_proba(Xe[:, M[mn][2]]), M[mn][1]) for mn in modlist}
            parts[c]['late_mean'].append(_combine([P[m] for m in modlist], None, 'mean'))
            parts[c]['evidential'].append(_combine([P[m] for m in modlist], [M[m][3] for m in modlist], 'evidential'))
            parts[c]['evidential_brier'].append(_combine([P[m] for m in modlist], [M[m][4] for m in modlist], 'evidential'))
    y_all = np.concatenate(Ys)
    s_all = np.concatenate(Ss)
    res = {}
    for c in conds:
        res[c] = {}
        pred = {}
        for op in ops:
            Pc = np.concatenate(parts[c][op])
            m = _prob_metrics(Pc, y_all)
            pr = Pc.argmax(1)
            res[c][op] = {'bacc': m['bacc'], 'brier': m['brier'], 'ece': m['ece'], 'recall_1': round(float((pr[y_all == 1] == 1).mean()), 3), 'recall_0': round(float((pr[y_all == 0] == 0).mean()), 3), 'mean_p1': round(float(Pc[:, 1].mean()), 3)}
            pred[op] = Pc.argmax(1)
        res[c]['paired_tests_bacc'] = {f'{a}_minus_{b}': _paired_bacc(pred[a], pred[b], y_all, s_all) for a, b in [('evidential', 'early'), ('early_dropout', 'early'), ('evidential', 'early_dropout'), ('late_mean', 'early')]}
    return res

def run4_case(dataset, tname, seeds):
    t0 = time.time()
    b = load_bundle(dataset)
    yt, mask = b['targets'][tname]
    yt = np.asarray(yt).astype(int)
    segs, build = load_clean(dataset)
    idx = np.flatnonzero(mask)
    Xclean = np.asarray(build([segs[i] for i in idx]), float)
    assert np.allclose(Xclean, b['X'][mask]), 'clean rebuild must equal load_bundle features'
    lb = LOAD_BEARING[dataset, tname]
    Xc = corrupted_features(segs, build, idx, LB_CHANNEL[lb])
    out = {'load_bearing': lb, 'n_windows': int(len(idx))}
    for s in seeds:
        out[f'seed{s}'] = corruption_eval(b['X'][mask], Xc, yt, b['si'][mask], b['mods'], seed=s)
    print(f'[run4] {dataset} {tname} (corrupt {lb}) done in {time.time() - t0:.0f}s', flush=True)
    return (dataset, tname, out)
POSTCAL_OPS = [('mean', 'late_mean', 'ba'), ('evidential', 'evidential', 'ba'), ('evidential', 'evidential_brier', 'brier')]

def postcal_eval(X, Xc, y, subj, mods, dominant, k=5, seed=0):
    modlist = [mn for mn, _ in mods]
    keep = [m for m in modlist if m != dominant]
    rng = np.random.default_rng(seed)
    conds = ['clean', f'drop:{dominant}'] + [c[0] for c in CORRUPTIONS]
    parts = {c: {name: {'raw': [], 'post': []} for _, name, _ in POSTCAL_OPS} for c in conds}
    Ys = []
    subj = np.asarray(subj)
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, y, rng, 2)
        Ys.append(y[te])
        prior1 = float(np.mean(y[fit_i]))
        M = {}
        for mn, ci in mods:
            clf, T, pc = _fit_cal(X[:, ci], y, fit_i, cal_i)
            M[mn] = (clf, T, ci, pc, _reliability(pc, y[cal_i]), brier_skill_reliability(pc, y[cal_i], prior1))

        def probs(Xe, mn):
            clf, T, ci = (M[mn][0], M[mn][1], M[mn][2])
            return _temper(clf.predict_proba(Xe[:, ci]), T)
        for how, name, rk in POSTCAL_OPS:
            w = (lambda m: M[m][4]) if rk == 'ba' else lambda m: M[m][5]
            f_cal = _combine([M[m][3] for m in modlist], [w(m) for m in modlist], how)
            Tp = _fit_temperature(f_cal, y[cal_i], [0, 1])
            for c in conds:
                if c == 'clean':
                    f = _combine([probs(X[te], m) for m in modlist], [w(m) for m in modlist], how)
                elif c.startswith('drop:'):
                    f = _combine([probs(X[te], m) for m in keep], [w(m) for m in keep], how)
                else:
                    f = _combine([probs(Xc[c][te], m) for m in modlist], [w(m) for m in modlist], how)
                parts[c][name]['raw'].append(f)
                parts[c][name]['post'].append(_temper(f, Tp))
    yy = np.concatenate(Ys)
    res = {}
    for c in conds:
        res[c] = {}
        for _, name, _ in POSTCAL_OPS:
            r = _prob_metrics(np.concatenate(parts[c][name]['raw']), yy)
            q = _prob_metrics(np.concatenate(parts[c][name]['post']), yy)
            assert r['bacc'] == q['bacc'], 'temperature must preserve the argmax'
            res[c][name] = {'bacc': r['bacc'], 'brier_raw': r['brier'], 'ece_raw': r['ece'], 'brier_frozen_postcal': q['brier'], 'ece_frozen_postcal': q['ece']}
    return res

def run6_case(dataset, tname, seeds):
    t0 = time.time()
    b = load_bundle(dataset)
    yt, mask = b['targets'][tname]
    yt = np.asarray(yt).astype(int)
    segs, build = load_clean(dataset)
    idx = np.flatnonzero(mask)
    lb = LOAD_BEARING[dataset, tname]
    Xc = corrupted_features(segs, build, idx, LB_CHANNEL[lb])
    out = {'load_bearing': lb}
    for s in seeds:
        out[f'seed{s}'] = postcal_eval(b['X'][mask], Xc, yt, b['si'][mask], b['mods'], lb, seed=s)
    print(f'[run6] {dataset} {tname} done in {time.time() - t0:.0f}s', flush=True)
    return (dataset, tname, out)

def _write(name, analysis, protocol, results):
    path = RESULTS / f'partitions_{name}.json'
    path.write_text(json.dumps({'analysis': analysis, 'protocol': protocol, 'results': results}, indent=2))
    print(f'wrote {path.relative_to(ROOT)}', flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?', choices=['run1', 'run3', 'run4', 'run5', 'run6'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--datasets', nargs='+', default=['ai4pain2026', 'ai4pain2025', 'biovid'])
    ap.add_argument('--jobs', type=int, default=5)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    sfx = f'_{a.tag}' if a.tag else ''
    if a.run == 'run1':
        jobs = [(ds, s) for s in a.seeds for ds in a.datasets]
        res = Parallel(n_jobs=a.jobs)((delayed(run1_one)(ds, s) for ds, s in jobs))
        out = {}
        for ds, s, r in res:
            out.setdefault(ds, {})[f'seed{s}'] = r
        _write('run1' + sfx, 'Run 1 + 2: repeated subject partitions; fusion value, operators, missing-modality fusion with dropout-trained early fusion, k-NN imputation and Brier-skill reliability', f'subject-grouped 5-fold, partition seeds {a.seeds}; subject-grouped calibration split; point estimates (n_perm=0); dropout p=0.25 x3 copies + availability indicators; kNN k=10; Brier-skill reliability on calibration subjects', out)
    elif a.run == 'run3':
        targets = {'ai4pain2026': ['presence', 'location'], 'ai4pain2025': ['presence', 'intensity'], 'biovid': ['presence', 'intensity']}
        jobs = [(ds, t, p) for ds in a.datasets for t in targets[ds] for p in PREDICTORS]
        res = Parallel(n_jobs=a.jobs)((delayed(run3_one)(*j) for j in jobs))
        out = {}
        for ds, t, p, r in res:
            out.setdefault(ds, {}).setdefault(t, {})[p] = r
        _write('run3' + sfx, 'Run 3: sensitivity of I_V to predictor and calibration', 'seed-0 subject-grouped 5-fold; all features; each fitted model scored without and with temperature scaling (T on subject-disjoint calibration split)', out)
    elif a.run == 'run4':
        cases = [(ds, t) for ds, t in LOAD_BEARING if ds in a.datasets]
        res = Parallel(n_jobs=a.jobs)((delayed(run4_case)(ds, t, a.seeds) for ds, t in cases))
        out = {}
        for ds, t, r in res:
            out.setdefault(ds, {})[t] = r
        _write('run4' + sfx, 'Run 4: partial degradation of the load-bearing channel at test time', f'subject-grouped 5-fold, partition seeds {a.seeds}; models fit on clean data; load-bearing channel of the test windows corrupted after normalization: Gaussian noise at SNR 10/5/0 dB (power = clean-window variance) or one 50% contiguous gap set to 0; fixed seed per (corruption, window); no operator told which channel is corrupted', out)
    elif a.run == 'run6':
        cases = [(ds, t) for ds, t in LOAD_BEARING if ds in a.datasets]
        res = Parallel(n_jobs=a.jobs)((delayed(run6_case)(ds, t, a.seeds) for ds, t in cases))
        out = {}
        for ds, t, r in res:
            out.setdefault(ds, {})[t] = r
        _write('run6' + sfx, 'Run 6: frozen post-fusion calibration under known loss and unflagged degradation (late and evidential fusion)', f'partition seeds {a.seeds}; post-fusion temperature fit on clean calibration subjects of each fold and frozen; applied to clean, known-loss (vote dropped) and corrupted test windows; bacc unchanged by construction', out)
    elif a.run == 'run5':
        jobs = [(ds, n) for ds in a.datasets for n in ('subject_all', 'baseline_only')]
        res = Parallel(n_jobs=a.jobs)((delayed(run5_one)(*j) for j in jobs))
        out = {}
        for ds, n, r in res:
            out.setdefault(ds, {})[n] = r
        _write('run5' + sfx, 'Run 5: prospective (baseline-only) normalization vs current', 'seed-0 subject-grouped 5-fold; AI4Pain: z-score from Baseline windows 1-2 (2026) or the first two 1022-sample Baseline windows (2025), those windows removed from BOTH variants; BioVid baseline_only = no subject statistics', out)
    else:
        ap.error('give a run')
if __name__ == '__main__':
    main()


class ModalityDropout:
    """Masks and availability indicators for dropout-trained early fusion."""
    masks = staticmethod(dropout_masks)
    apply = staticmethod(_apply_avail)
    brier_skill_reliability = staticmethod(brier_skill_reliability)


class SignalCorruptor:
    """Silent degradation: additive Gaussian noise at a target SNR or a contiguous gap."""
    corrupt = staticmethod(corrupt_signal)
    corrupted_features = staticmethod(corrupted_features)
