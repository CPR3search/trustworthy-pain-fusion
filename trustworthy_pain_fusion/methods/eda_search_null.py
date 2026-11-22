"""Permutation null that repeats the complete representation search."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import argparse
import json
import time
from functools import partial
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score
from trustworthy_pain_fusion.methods.eda_search import CACHE, _lgbm, _pipe, _task
from trustworthy_pain_fusion.methods.splits import loso_splits
RESULTS = _PROJECT_ROOT / 'results'
FAMILIES_JSON = RESULTS / 'eda_localise_families.json'
OBS_JSON = RESULTS / 'eda_localise_matched_observed.json'
NULL_JSON = {'within': RESULTS / 'eda_localise_nullmax_loso_matched_within.json'}
CKPT = {'within': _PROJECT_ROOT / 'cache' / 'eda_localise_nullmax_loso_matched_within_ckpt.npz'}
SCHEME_TEXT = {'within': 'numpy default_rng(0); labels permuted within each subject per draw'}
SEEDED_MI = partial(mutual_info_classif, random_state=0)

def _seed_mi(pipe):
    if 'sel' in pipe.named_steps:
        pipe.set_params(sel__score_func=SEEDED_MI)
    return pipe

def search_configs(fam, group):
    cfgs = [('logreg', None, _pipe('logreg')), ('lda_mi30', None, _pipe('lda', k=30)), ('svm_mi40', None, _pipe('svm', k=40)), ('rf', None, _pipe('rf')), ('lgbm', None, _lgbm())]
    cfgs += [(f'lgbm_mi{k}', None, _lgbm(k=k)) for k in (10, 20, 40, 80)]
    cfgs += [(f'group_{g}', np.where(group == g)[0], _lgbm()) for g in ('eda', 'bp')]
    cfgs += [(f'family_{f}', np.where(fam == f)[0], _lgbm()) for f in dict.fromkeys(fam)]
    return [(str(n), c, _seed_mi(e)) for n, c, e in cfgs]

def _config_score(X, y, splits, cols, proto):
    Xc = X if cols is None else X[:, cols]
    accs = [balanced_accuracy_score(y[te], clone(proto).fit(Xc[tr], y[tr]).predict(Xc[te])) for tr, te in splits]
    return float(np.mean(accs))

def _timed_score(X, y, splits, cols, proto):
    t = time.time()
    return (_config_score(X, y, splits, cols, proto), time.time() - t)

def _all_scores(X, y, splits, cfgs):
    return [_config_score(X, y, splits, c, p) for _, c, p in cfgs]

def _permute(y, si, rng, scheme):
    yp = y.copy()
    for s in np.unique(si):
        idx = np.flatnonzero(si == s)
        yp[idx] = y[idx][rng.permutation(len(idx))]
    return yp

def _load():
    d = np.load(CACHE, allow_pickle=True)
    X, lab, si, fam, group = (d['X'], d['lab'], d['si'], d['fam'], d['group'])
    y, mask = _task(lab, 'location')
    return ((np.ascontiguousarray(X[mask]), y, si[mask], fam, group), (X, lab, si))

def observed(n_jobs):
    (X, y, si, fam, group), (Xall, lab, siall) = _load()
    cfgs = search_configs(fam, group)
    names = [n for n, _, _ in cfgs]
    splits = list(loso_splits(si))
    yd, md = _task(lab, 'detection')
    Xd, splits_d = (np.ascontiguousarray(Xall[md]), list(loso_splits(siall[md])))
    t0 = time.time()
    jobs = [delayed(_timed_score)(X, y, splits, c, p) for _, c, p in cfgs]
    jobs.append(delayed(_timed_score)(Xd, yd, splits_d, None, _seed_mi(_pipe('svm', k=40))))
    res = Parallel(n_jobs=n_jobs, verbose=5)(jobs)
    (ctrl, _), res = (res[-1], res[:-1])
    scores = {n: s for n, (s, _) in zip(names, res)}
    secs = {n: round(t, 1) for n, (_, t) in zip(names, res)}
    ref = json.loads(FAMILIES_JSON.read_text())
    repro = {}
    for n in names:
        if n.startswith('family_'):
            exp = ref['families'][n[len('family_'):]]['location']
        elif n.startswith('group_'):
            exp = ref['groups'][n[len('group_'):]]['location']
        else:
            continue
        repro[n] = {'expected': exp, 'got': round(scores[n] * 100, 2)}
    max_dev = max((abs(v['expected'] - v['got']) for v in repro.values()))
    ok = max_dev <= 0.011
    best = max(scores, key=scores.get)
    out = {'task': 'location (arm vs hand, true-pain segments), LOSO balanced accuracy', 'n_configs': len(names), 'scores_raw': scores, 'scores_pct': {n: round(s * 100, 2) for n, s in scores.items()}, 'best_config': best, 'best_pct': round(scores[best] * 100, 2), 'reproduction_vs_families_json': repro, 'reproduction_max_abs_dev': round(max_dev, 3), 'reproduction_ok': bool(ok), 'detection_control_svm_mi40_pct': round(ctrl * 100, 2), 'seconds_per_config': secs, 'one_search_cpu_seconds': round(sum(secs.values()), 1), 'wall_seconds': round(time.time() - t0, 1)}
    OBS_JSON.write_text(json.dumps(out, indent=2))
    print(f"best-over-search {out['best_pct']} ({best}); reproduction max |dev| {out['reproduction_max_abs_dev']} -> {('OK' if ok else 'FAILED')}; detection control {out['detection_control_svm_mi40_pct']}; one search ~{out['one_search_cpu_seconds']} CPU-s; wrote {OBS_JSON.name}")
    return 0 if ok else 3

def null(n_perm, n_jobs, chunk, scheme):
    rec = json.loads(OBS_JSON.read_text())
    if not rec['reproduction_ok']:
        raise SystemExit("observed pass did not reproduce the figure's values; null not run")
    (X, y, si, fam, group), _ = _load()
    cfgs = search_configs(fam, group)
    names = [n for n, _, _ in cfgs]
    if names != list(rec['scores_raw']):
        raise SystemExit('configuration set changed since the observed pass')
    obs = np.array([rec['scores_raw'][n] for n in names])
    splits = list(loso_splits(si))
    rng = np.random.default_rng(0)
    perms = [_permute(y, si, rng, scheme) for _ in range(n_perm)]
    ckpt = CKPT[scheme]
    done = np.empty((0, len(names)))
    if ckpt.exists():
        ck = np.load(ckpt, allow_pickle=True)
        if list(ck['names']) != names or str(ck['scheme']) != scheme:
            raise SystemExit('checkpoint was written for a different configuration set or scheme')
        done = ck['null_scores']
        print(f'resuming from checkpoint: {len(done)} draws done', flush=True)
    for start in range(len(done), n_perm, chunk):
        stop = min(start + chunk, n_perm)
        t = time.time()
        rows = Parallel(n_jobs=n_jobs)((delayed(_all_scores)(X, perms[i], splits, cfgs) for i in range(start, stop)))
        done = np.vstack([done, np.asarray(rows)])
        np.savez(ckpt, null_scores=done, names=np.array(names), scheme=np.array(scheme))
        print(f"[{time.strftime('%H:%M:%S')}] draws {stop}/{n_perm} ({time.time() - t:.0f}s this chunk)", flush=True)
    D = done[:n_perm]
    null_max = D.max(axis=1)
    p = (np.sum(null_max >= obs.max()) + 1) / (len(null_max) + 1)
    counts = np.bincount(D.argmax(axis=1), minlength=len(names))
    out = {'observed_best_loso': round(float(obs.max()), 4), 'observed_best_config': names[int(obs.argmax())], 'null_mean': round(float(null_max.mean()), 4), 'null_std': round(float(null_max.std()), 4), 'null_p95': round(float(np.percentile(null_max, 95)), 4), 'corrected_p': round(float(p), 4), 'n_perm': int(n_perm), 'n_configs': len(names), 'configs': names, 'perm_scheme': scheme, 'perm_rng': SCHEME_TEXT[scheme], 'observed_scores': {n: round(float(s), 4) for n, s in zip(names, obs)}, 'per_config_null_mean': {n: round(float(D[:, i].mean()), 4) for i, n in enumerate(names)}, 'null_max_config_counts': {n: int(c) for n, c in zip(names, counts) if c}}
    NULL_JSON[scheme].write_text(json.dumps(out, indent=2))
    print(f'\nSEARCH-MATCHED LOSO max-statistic null ({n_perm} draws, {len(names)} configs, {scheme} permutation):')
    print(f"  observed best-over-search = {obs.max() * 100:.2f} ({out['observed_best_config']})")
    print(f'  null-of-max = {null_max.mean() * 100:.2f} +/- {null_max.std() * 100:.2f} [95th pct {np.percentile(null_max, 95) * 100:.2f}]')
    print(f'  corrected p = {p:.4f}; wrote {NULL_JSON[scheme].name}')
    if 0.01 < p < 0.1 and n_perm < 1000:
        print('  PRE-DECLARED RULE: 0.01 < p < 0.10 -> extend to 1000 draws (rerun with --n-perm 1000; resumes from checkpoint)')

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('observed')
    o.add_argument('--n-jobs', type=int, default=16)
    n = sub.add_parser('null')
    n.add_argument('--n-perm', type=int, default=1000)
    n.add_argument('--n-jobs', type=int, default=16)
    n.add_argument('--chunk', type=int, default=None)
    n.add_argument('--scheme', choices=['within'], default='within')
    a = ap.parse_args()
    if a.cmd == 'observed':
        raise SystemExit(observed(a.n_jobs))
    null(a.n_perm, a.n_jobs, a.chunk or a.n_jobs, a.scheme)
if __name__ == '__main__':
    main()
