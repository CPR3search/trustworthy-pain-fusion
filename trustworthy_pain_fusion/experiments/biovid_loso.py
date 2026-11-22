"""Leave-one-subject-out evaluation on BioVid Part A."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.data import biovid
from trustworthy_pain_fusion.methods.splits import loso_splits
ROOT = _PROJECT_ROOT

def _loso_acc(X, y, subjects):
    y = np.asarray(y).astype(int)
    subjects = np.asarray(subjects)
    accs = []
    for tr, te in loso_splits(subjects):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(float((clf.predict(sc.transform(X[te])) == y[te]).mean()))
    return (float(np.mean(accs)), float(np.std(accs)), len(accs))

def _eval_subset(X, y, subj, label):
    m = (y == 0) | (y == 4)
    a, s, n = _loso_acc(X[m], (y[m] == 4).astype(int), subj[m])
    a5, s5, n5 = _loso_acc(X, y, subj)
    print(f'[{label}] n_subj={len(set(subj))} | presence BL1-vs-PA4 {a:.3f}+/-{s:.3f} (n={n}) | intensity 5-cls {a5:.3f}+/-{s5:.3f} (n={n5})', flush=True)
    return {'n_subjects': int(len(set(subj))), 'presence_BL1_vs_PA4': {'acc': round(a, 3), 'std': round(s, 3), 'n_folds': n, 'chance': 0.5}, 'intensity_5class': {'acc': round(a5, 3), 'std': round(s5, 3), 'n_folds': n5, 'chance': 0.2}}

def main():
    print('loading BioVid Part A (8700 trials)...', flush=True)
    segs, subj, lab, _ = biovid.load_segments()
    segs = biovid.normalize_per_subject(segs, subj)
    X = biovid.build_features(segs)
    y = np.asarray(lab, int)
    subj = np.asarray(subj)
    print(f'X={X.shape}  n_subj={len(set(subj))}', flush=True)
    out = {'dataset': 'BioVid Part A (biosignals_filtered)', 'protocol': 'LOSO (leave-one-subject-out), per-subject z-norm, logreg', 'n_features': int(X.shape[1]), 'channels_with_signal': ['gsr', 'ecg', 'emg_trapezius'], 'subsets': {}}
    out['subsets']['all_87'] = _eval_subset(X, y, subj, 'all-87')
    excl = set(biovid.weak_responders())
    keep = ~np.isin(subj, list(excl))
    n_excl = len(set(subj) & excl)
    out['subsets']['reduced_67'] = _eval_subset(X[keep], y[keep], subj[keep], 'reduced-67')
    out['subsets']['reduced_67']['excluded_n'] = n_excl
    out['subsets']['reduced_67']['cite'] = 'werner2017expressiveness (ACIIW 2017)'
    dst = ROOT / 'results' / 'biovid_loso.json'
    dst.write_text(json.dumps(out, indent=2))
    print('wrote', dst, flush=True)
if __name__ == '__main__':
    main()
