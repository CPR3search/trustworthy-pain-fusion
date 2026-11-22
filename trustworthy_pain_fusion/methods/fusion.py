"""Early, late and evidential fusion; complete and missing inputs."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from trustworthy_pain_fusion.methods.splits import grouped_kfold
from trustworthy_pain_fusion.methods.decodability import _entropy_bits, _temper, _ce_bits, _fit_temperature, _lgbm, inner_split, decodability_profile
ROOT = _PROJECT_ROOT
RESULTS = ROOT / 'results'
SEED = 0
COMBINERS = [('mean', 'late_mean'), ('product', 'late_product'), ('evidential', 'evidential')]

def _mod_cols(names, prefix):
    return [i for i, n in enumerate(names) if n == prefix or n.startswith(prefix + '_')]

def load_bundle(dataset):
    if dataset == 'ai4pain2026':
        from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
        segs, subj, lab, _ = ai4pain.load_segments('data', 'train', ai4pain.SIGNALS)
        segs = ai4pain.normalize_per_subject(segs, subj, ai4pain.SIGNALS)
        X = ai4pain.build_features(segs, ai4pain.SIGNALS)
        names = ai4pain.feature_names(ai4pain.SIGNALS)
        mod_names = ['Bvp', 'Eda', 'Resp', 'SpO2']
        y = np.asarray(lab, int)
        pain = y != 0
        ones = np.ones(len(y), bool)
        targets = {'presence': ((y != 0).astype(int), ones), 'location': ((y[pain] == 2).astype(int), pain)}
    elif dataset == 'ai4pain2025':
        from trustworthy_pain_fusion.data import ai4pain2025 as a25
        segs, subj, lab, _ = a25.load_segments(split='train', nopain_mode='baseline_seg')
        segs = a25.normalize_per_subject(segs, subj)
        X = a25.build_features(segs)
        names = a25.feature_names()
        mod_names = ['Bvp', 'Eda', 'Resp', 'SpO2']
        y = np.asarray(lab, int)
        pain = y != 0
        ones = np.ones(len(y), bool)
        targets = {'presence': ((y != 0).astype(int), ones), 'intensity': ((y[pain] == 2).astype(int), pain)}
    elif dataset == 'biovid':
        from trustworthy_pain_fusion.data import biovid
        segs, subj, lab, _ = biovid.load_segments()
        segs = biovid.normalize_per_subject(segs, subj)
        X = biovid.build_features(segs)
        names = biovid.feature_names()
        mod_names = ['gsr', 'ecg', 'emg_trapezius']
        y = np.asarray(lab, int)
        m = (y == 0) | (y == 4)
        ones = np.ones(len(y), bool)
        targets = {'presence': ((y[m] == 4).astype(int), m), 'intensity': (y.astype(int), ones)}
    else:
        raise ValueError(dataset)
    mods = [(mn, _mod_cols(names, mn)) for mn in mod_names]
    mods = [(mn, ci) for mn, ci in mods if ci]
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    return {'X': np.asarray(X, float), 'names': names, 'mods': mods, 'si': si, 'targets': targets}

def _inner(tr, subj, y, rng, n_classes, tries=10):
    for _ in range(tries):
        fit_i, cal_i = inner_split(tr, subj, rng)
        if len(cal_i) and len(np.unique(y[fit_i])) >= n_classes and (len(np.unique(y[cal_i])) >= min(2, n_classes)):
            return (fit_i, cal_i)
    tr = np.asarray(tr)
    return (tr, tr)

def _iv_on(X, y, subj, k=5, seed=SEED):
    return decodability_profile(X, y, subj, k=k, seed=seed, n_perm=0, n_jobs=1)

def _iv_summary(p):
    return {'iv_bits': p['usable_info_bits'], 'iv_unclamped': p.get('usable_info_unclamped_bits', p['usable_info_bits']), 'acc': p['cv_accuracy'], 'bacc': p['cv_balanced_accuracy']}

def fusion_value(datasets):
    out = {}
    for ds in datasets:
        b = load_bundle(ds)
        out[ds] = {}
        for tname, (yt, mask) in b['targets'].items():
            subj = b['si'][mask]
            per_mod = {mn: _iv_summary(_iv_on(b['X'][mask][:, ci], yt, subj)) for mn, ci in b['mods']}
            fused_cols = sorted({i for _, ci in b['mods'] for i in ci})
            fused = _iv_summary(_iv_on(b['X'][mask][:, fused_cols], yt, subj))
            best_mod = max(per_mod, key=lambda m: per_mod[m]['iv_bits'])
            best_iv = per_mod[best_mod]['iv_bits']
            out[ds][tname] = {'per_modality': per_mod, 'fused': fused, 'best_single_modality': best_mod, 'best_single_iv': best_iv, 'fusion_gain_bits': round(fused['iv_bits'] - best_iv, 3)}
            print(f"[N1] {ds:12s} {tname:9s} best={best_mod}({best_iv:.3f}) fused={fused['iv_bits']:.3f} gain={fused['iv_bits'] - best_iv:+.3f}", flush=True)
    return out

def missing_modality(datasets):
    out = {}
    for ds in datasets:
        b = load_bundle(ds)
        out[ds] = {}
        for tname, (yt, mask) in b['targets'].items():
            subj = b['si'][mask]
            Xt = b['X'][mask]
            fused_cols = sorted({i for _, ci in b['mods'] for i in ci})
            full = _iv_summary(_iv_on(Xt[:, fused_cols], yt, subj))
            drop_test, lomo = ({}, {})
            for mn, ci in b['mods']:
                d_iv, d_acc, d_bacc = _impute_drop_iv(Xt, yt, subj, fused_cols, ci)
                drop_test[mn] = {'iv_bits': d_iv, 'acc': d_acc, 'bacc': d_bacc, 'acc_drop': round(full['acc'] - d_acc, 3)}
                keep = [i for i in fused_cols if i not in set(ci)]
                lo = _iv_summary(_iv_on(Xt[:, keep], yt, subj))
                lomo[mn] = {**lo, 'acc_drop': round(full['acc'] - lo['acc'], 3)}
                print(f"[N2] {ds:12s} {tname:9s} -{mn:13s} testdrop_acc={d_acc:.3f} lomo_acc={lo['acc']:.3f}", flush=True)
            out[ds][tname] = {'full': full, 'test_time_drop': drop_test, 'leave_one_modality_out': lomo}
    return out

def _impute_drop_iv(X, y, subj, fused_cols, drop_cols, k=5, seed=SEED):
    y = np.asarray(y).astype(int)
    ncls = int(len(np.unique(y)))
    H = _entropy_bits(y)
    drop = set(drop_cols)
    rng = np.random.default_rng(seed)
    Xf = X[:, fused_cols]
    drop_local = [j for j, c in enumerate(fused_cols) if c in drop]
    ces, accs, Y, P = ([], [], [], [])
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, y, rng, ncls)
        clf = _lgbm().fit(Xf[fit_i], y[fit_i])
        classes = list(clf.classes_)
        mean = Xf[fit_i].mean(0)
        Xcal = Xf[cal_i].copy()
        Xcal[:, drop_local] = mean[drop_local]
        Xte = Xf[te].copy()
        Xte[:, drop_local] = mean[drop_local]
        T = _fit_temperature(clf.predict_proba(Xcal), y[cal_i], classes)
        proba = _temper(clf.predict_proba(Xte), T)
        ces.append(_ce_bits(proba, y[te], classes))
        pred = clf.predict(Xte)
        accs.append(float((pred == y[te]).mean()))
        Y.append(y[te])
        P.append(pred)
    bacc = balanced_accuracy_score(np.concatenate(Y), np.concatenate(P))
    return (round(max(0.0, H - float(np.mean(ces))), 3), round(float(np.mean(accs)), 3), round(float(bacc), 3))

def _ece(conf, correct, n_bins=15):
    conf = np.asarray(conf)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)

def _prob_metrics(P, y):
    pred = P.argmax(1)
    conf = P.max(1)
    correct = pred == y
    idx = np.arange(len(y))
    return {'acc': round(float(correct.mean()), 3), 'bacc': round(float(balanced_accuracy_score(y, pred)), 3), 'ece': round(_ece(conf, correct), 3), 'brier': round(float(np.mean((P[idx, y] - 1) ** 2 + P[idx, 1 - y] ** 2)), 3)}

def _reliability(p_cal, y_cal):
    if len(np.unique(y_cal)) < 2:
        return 0.0
    return float(np.clip(2 * balanced_accuracy_score(y_cal, p_cal.argmax(1)) - 1, 0.0, 1.0))

def _ds_combine(probas, rel):
    N = probas[0].shape[0]
    a0 = np.zeros(N)
    a1 = np.zeros(N)
    aT = np.ones(N)
    keep = np.ones(N)
    for p, r in zip(probas, rel):
        b0 = r * p[:, 0]
        b1 = r * p[:, 1]
        bT = 1.0 - r
        K = a0 * b1 + a1 * b0
        denom = np.clip(1.0 - K, 1e-09, None)
        a0, a1, aT = ((a0 * b0 + a0 * bT + aT * b0) / denom, (a1 * b1 + a1 * bT + aT * b1) / denom, aT * bT / denom)
        keep = keep * (1.0 - K)
    p0 = a0 + aT / 2.0
    p1 = a1 + aT / 2.0
    s = np.clip(p0 + p1, 1e-09, None)
    return (np.stack([p0 / s, p1 / s], 1), 1.0 - keep)

def _combine(probas, rel, how):
    stack = np.stack(probas, 0)
    if how == 'mean':
        return stack.mean(0)
    if how == 'product':
        lp = np.log(np.clip(stack, 1e-09, 1)).sum(0)
        p = np.exp(lp - lp.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)
    if how == 'evidential':
        return _ds_combine(probas, rel)[0]
    raise ValueError(how)

def _cal_proba_pair(X, y, fit_i, cal_i, eval_i):
    clf = _lgbm().fit(X[fit_i], y[fit_i])
    classes = list(clf.classes_)
    p_cal = clf.predict_proba(X[cal_i])
    T = _fit_temperature(p_cal, y[cal_i], classes)
    return (_temper(p_cal, T), _temper(clf.predict_proba(X[eval_i]), T))

def _cal_proba(X, y, fit_i, cal_i, eval_i):
    return _cal_proba_pair(X, y, fit_i, cal_i, eval_i)[1]

def fusion_methods(datasets):
    out = {}
    for ds in datasets:
        b = load_bundle(ds)
        out[ds] = {}
        for tname, (yt, mask) in b['targets'].items():
            yt = np.asarray(yt).astype(int)
            if len(np.unique(yt)) != 2:
                continue
            res = _fusion_methods_one(b['X'][mask], yt, b['si'][mask], b['mods'])
            out[ds][tname] = res
            print(f'[N3] {ds:12s} {tname:9s} ' + ' '.join((f"{k}:bacc={v['bacc']:.3f},ece={v['ece_postcal']:.3f}" for k, v in res.items())), flush=True)
    return out

def _fusion_methods_one(X, y, subj, mods, k=5, seed=SEED):
    fused_cols = sorted({i for _, ci in mods for i in ci})
    rng = np.random.default_rng(seed)
    ops = ['early'] + [name for _, name in COMBINERS]
    agg = {s: {'raw': [], 'post': [], 'K': []} for s in ops}
    Ys = []
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, y, rng, 2)
        early = _cal_proba(X[:, fused_cols], y, fit_i, cal_i, te)
        agg['early']['raw'].append(early)
        agg['early']['post'].append(early)
        p_cal, p_te, rel = ([], [], [])
        for _, ci in mods:
            pc, pt = _cal_proba_pair(X[:, ci], y, fit_i, cal_i, te)
            p_cal.append(pc)
            p_te.append(pt)
            rel.append(_reliability(pc, y[cal_i]))
        for how, name in COMBINERS:
            f_te = _combine(p_te, rel, how)
            f_cal = _combine(p_cal, rel, how)
            T = _fit_temperature(f_cal, y[cal_i], [0, 1])
            agg[name]['raw'].append(f_te)
            agg[name]['post'].append(_temper(f_te, T))
        agg['evidential']['K'].append(_ds_combine(p_te, rel)[1])
        Ys.append(y[te])
    yy = np.concatenate(Ys)
    res = {}
    for s in ops:
        raw = _prob_metrics(np.concatenate(agg[s]['raw']), yy)
        post = _prob_metrics(np.concatenate(agg[s]['post']), yy)
        res[s] = {**raw, 'ece_postcal': post['ece'], 'brier_postcal': post['brier']}
        if s == 'evidential':
            res[s]['mean_conflict_K'] = round(float(np.concatenate(agg[s]['K']).mean()), 3)
            res[s]['conflict_definition'] = 'total conflict 1 - prod_i(1 - K_i)'
    return res

def missing_modality_fusion(datasets, fusion_value_results=None):
    if fusion_value_results is None and (RESULTS / 'fusion_value.json').exists():
        fusion_value_results = json.loads((RESULTS / 'fusion_value.json').read_text())['results']
    out = {}
    for ds in datasets:
        b = load_bundle(ds)
        out[ds] = {}
        for tname, (yt, mask) in b['targets'].items():
            yt = np.asarray(yt).astype(int)
            if len(np.unique(yt)) != 2:
                continue
            dominant = None
            fv = (fusion_value_results or {}).get(ds, {}).get(tname)
            if fv:
                pm = fv['per_modality']
                dominant = max(pm, key=lambda m: pm[m]['iv_bits'])
            res = _missing_fusion_one(b['X'][mask], yt, b['si'][mask], b['mods'], dominant=dominant)
            out[ds][tname] = res
            full = res['none']
            print(f"[N4] {ds:12s} {tname:9s} full early={full['early']['bacc']:.3f} evid={full['evidential']['bacc']:.3f} | dominant={dominant}", flush=True)
    return out

def _paired(correct_a, correct_b, subj, n_boot=2000, seed=SEED):
    subs = np.unique(subj)
    diff = np.array([correct_a[subj == s].mean() - correct_b[subj == s].mean() for s in subs])
    return _paired_stats(diff, n_boot, seed)

def _subject_bacc(pred, y, subj, s):
    m = subj == s
    if len(np.unique(y[m])) < 2:
        return float((pred[m] == y[m]).mean())
    return float(balanced_accuracy_score(y[m], pred[m]))

def _paired_bacc(pred_a, pred_b, y, subj, n_boot=2000, seed=SEED):
    subs = np.unique(subj)
    diff = np.array([_subject_bacc(pred_a, y, subj, s) - _subject_bacc(pred_b, y, subj, s) for s in subs])
    return _paired_stats(diff, n_boot, seed)

def _paired_stats(diff, n_boot=2000, seed=SEED):
    diff = np.round(np.asarray(diff, dtype=float), 10)
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(n_boot)])
    try:
        p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = 1.0
    return {'mean_diff': round(float(diff.mean()), 4), 'ci95': [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)], 'wilcoxon_p': float(f'{p:.3g}'), 'n_subjects': int(len(diff))}

def _missing_fusion_one(X, y, subj, mods, k=5, seed=SEED, dominant=None):
    fused_cols = sorted({i for _, ci in mods for i in ci})
    modlist = [mn for mn, _ in mods]
    rng = np.random.default_rng(seed)
    scen = ['none'] + [f'drop:{m}' for m in modlist]
    ops = {s: (['early'] if s == 'none' else ['early_impute']) + [n for _, n in COMBINERS] for s in scen}
    parts = {s: {op: [] for op in ops[s]} for s in scen}
    Ys, Ss = ([], [])
    subj = np.asarray(subj)
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        fit_i, cal_i = _inner(tr, subj, y, rng, 2)
        Ys.append(y[te])
        Ss.append(subj[te])
        P, R = ({}, {})
        for mn, ci in mods:
            pc, pt = _cal_proba_pair(X[:, ci], y, fit_i, cal_i, te)
            P[mn] = pt
            R[mn] = _reliability(pc, y[cal_i])
        Xf = X[:, fused_cols]
        clf = _lgbm().fit(Xf[fit_i], y[fit_i])
        classes = list(clf.classes_)
        T = _fit_temperature(clf.predict_proba(Xf[cal_i]), y[cal_i], classes)
        parts['none']['early'].append(_temper(clf.predict_proba(Xf[te]), T))
        for how, name in COMBINERS:
            parts['none'][name].append(_combine([P[m] for m in modlist], [R[m] for m in modlist], how))
        mean = Xf[fit_i].mean(0)
        for drop, ci in mods:
            s = f'drop:{drop}'
            local = [j for j, c in enumerate(fused_cols) if c in set(ci)]
            Xte = Xf[te].copy()
            Xte[:, local] = mean[local]
            parts[s]['early_impute'].append(_temper(clf.predict_proba(Xte), T))
            keep = [m for m in modlist if m != drop]
            for how, name in COMBINERS:
                parts[s][name].append(_combine([P[m] for m in keep], [R[m] for m in keep], how))
    y_all = np.concatenate(Ys)
    s_all = np.concatenate(Ss)
    res = {}
    pred = {}
    for s in scen:
        res[s] = {}
        pred[s] = {}
        for op in ops[s]:
            Pc = np.concatenate(parts[s][op])
            m = _prob_metrics(Pc, y_all)
            res[s][op] = {'acc': m['acc'], 'bacc': m['bacc'], 'ece': m['ece']}
            pred[s][op] = Pc.argmax(1)
    pairs = [('evidential', 'early_impute'), ('late_mean', 'early_impute'), ('late_mean', 'evidential')]
    res['paired_tests'] = {s: {**{f'{a}_minus_{b}': _paired(pred[s][a] == y_all, pred[s][b] == y_all, s_all) for a, b in pairs}, **{f'{a}_minus_{b}_bacc': _paired_bacc(pred[s][a], pred[s][b], y_all, s_all) for a, b in pairs}} for s in scen if s != 'none'}
    res['dominant_modality'] = dominant
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['ai4pain2026', 'ai4pain2025', 'biovid'])
    ap.add_argument('--only', choices=['n1', 'n2', 'n3', 'n4'], default=None)
    args = ap.parse_args()
    v = None
    if args.only in (None, 'n1'):
        v = fusion_value(args.datasets)
        (RESULTS / 'fusion_value.json').write_text(json.dumps({'analysis': 'N1 fusion value (per-modality vs fused, calibrated I_V)', 'protocol': 'subject-grouped 5-fold; subject-grouped calibration split; temperature-calibrated V-information; balanced accuracy', 'results': v}, indent=2))
        print('wrote results/fusion_value.json', flush=True)
    if args.only in (None, 'n2'):
        m = missing_modality(args.datasets)
        (RESULTS / 'missing_modality.json').write_text(json.dumps({'analysis': 'N2 missing-modality robustness', 'protocol': 'subject-grouped 5-fold; subject-grouped calibration split; (a) test-time drop with fit-split-mean imputation, (b) leave-one-modality-out retrain', 'results': m}, indent=2))
        print('wrote results/missing_modality.json', flush=True)
    if args.only in (None, 'n3'):
        f = fusion_methods(args.datasets)
        (RESULTS / 'fusion_methods.json').write_text(json.dumps({'analysis': 'N3 fusion operators (early vs late vs evidential)', 'protocol': 'subject-grouped 5-fold; subject-grouped calibration split; per-modality temperature-calibrated posteriors; class-aware reliability (2*balanced accuracy - 1); late/evidential outputs also temperature-calibrated after fusion (ece_postcal, brier_postcal); binary targets; ECE 15-bin; total DS conflict', 'results': f}, indent=2))
        print('wrote results/fusion_methods.json', flush=True)
    if args.only in (None, 'n4'):
        mf = missing_modality_fusion(args.datasets, v)
        (RESULTS / 'missing_fusion.json').write_text(json.dumps({'analysis': 'N4 fusion under missing modalities (vote-drop vs impute)', 'protocol': 'subject-grouped 5-fold; subject-grouped calibration split; binary targets; early fusion imputes the missing modality to the fit-split mean, late/evidential drop its vote; reliability weights per fold; dominant modality = highest single-modality I_V (N1), fixed before the comparison; subject-paired Wilcoxon tests with bootstrap CIs', 'results': mf}, indent=2))
        print('wrote results/missing_fusion.json', flush=True)
if __name__ == '__main__':
    main()


class FusionOperators:
    """Late averaging, log-opinion product and Dempster-Shafer evidential fusion."""
    reliability = staticmethod(_reliability)
    dempster_shafer = staticmethod(_ds_combine)
    combine = staticmethod(_combine)
    calibrated_probabilities = staticmethod(_cal_proba)
