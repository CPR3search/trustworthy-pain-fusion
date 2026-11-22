"""Cross-dataset presence transfer on EDA/GSR features."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
from trustworthy_pain_fusion.methods.splits import grouped_kfold
from trustworthy_pain_fusion.data import biovid
from trustworthy_pain_fusion.data import ai4pain2025 as a25
ROOT = _PROJECT_ROOT
_SHARED = ai4pain._SHARED_NAMES + ai4pain._EDA_NAMES

def _eda_matrix(X, names, prefix):
    idx = []
    for s in _SHARED:
        col = f'{prefix}_{s}'
        idx.append(names.index(col))
    return X[:, idx]

def _load_presence(which):
    if which == 'ai4pain2026':
        segs, subj, lab, _ = ai4pain.load_segments('data', 'train', ai4pain.SIGNALS)
        segs = ai4pain.normalize_per_subject(segs, subj, ai4pain.SIGNALS)
        X = ai4pain.build_features(segs, ai4pain.SIGNALS)
        names = ai4pain.feature_names(ai4pain.SIGNALS)
        prefix = 'Eda'
        y = (np.asarray(lab, int) != 0).astype(int)
    elif which == 'ai4pain2025':
        segs, subj, lab, _ = a25.load_segments(split='train', nopain_mode='baseline_seg')
        segs = a25.normalize_per_subject(segs, subj)
        X = a25.build_features(segs)
        names = a25.feature_names()
        prefix = 'Eda'
        y = (np.asarray(lab, int) != 0).astype(int)
    elif which == 'biovid':
        segs, subj, lab, _ = biovid.load_segments()
        segs = biovid.normalize_per_subject(segs, subj)
        X = biovid.build_features(segs)
        names = biovid.feature_names()
        prefix = 'gsr'
        y0 = np.asarray(lab, int)
        m = (y0 == 0) | (y0 == 4)
        X, y, subj = (X[m], (y0[m] == 4).astype(int), list(np.asarray(subj)[m]))
    else:
        raise ValueError(which)
    return (_eda_matrix(X, names, prefix), np.asarray(y), np.asarray(subj))

def _fit_predict(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced').fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))

def _within(X, y, subj, k=5, seed=0):
    Y, P, accs = ([], [], [])
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        pred = _fit_predict(X[tr], y[tr], X[te])
        accs.append(float((pred == y[te]).mean()))
        Y.append(y[te])
        P.append(pred)
    return (float(np.mean(accs)), float(balanced_accuracy_score(np.concatenate(Y), np.concatenate(P))))

def _transfer(Xtr, ytr, Xte, yte):
    pred = _fit_predict(Xtr, ytr, Xte)
    return (float((pred == yte).mean()), float(balanced_accuracy_score(yte, pred)))

def main():
    names = ['ai4pain2026', 'ai4pain2025', 'biovid']
    data = {n: _load_presence(n) for n in names}
    majority = {}
    for n in names:
        X, y, s = data[n]
        majority[n] = round(float(max(y.mean(), 1 - y.mean())), 3)
        print(f'{n}: X={X.shape} pos_rate={y.mean():.2f} majority={majority[n]} n_subj={len(set(s))}', flush=True)
    out_matrix = {}
    for tr in names:
        Xtr, ytr, str_ = data[tr]
        out_matrix[tr] = {}
        for te in names:
            Xte, yte, _ = data[te]
            if tr == te:
                acc, bacc = _within(Xtr, ytr, str_)
                kind = 'within (5-fold)'
            else:
                acc, bacc = _transfer(Xtr, ytr, Xte, yte)
                kind = 'transfer'
            out_matrix[tr][te] = {'acc': round(acc, 3), 'bacc': round(bacc, 3), 'kind': kind}
            print(f'train {tr:12s} -> test {te:12s} : acc {acc:.3f} bal.acc {bacc:.3f} ({kind})', flush=True)
    out = {'task': 'pain presence (No-Pain vs Pain; BioVid BL1 vs PA4)', 'feature_space': 'shared EDA/GSR descriptor (37 features), identical across datasets', 'n_features': len(_SHARED), 'chance_balanced_accuracy': 0.5, 'majority_rate': majority, 'protocol': 'per-subject signal z-scoring within each dataset; feature scaler and class-balanced logistic regression fit on the training dataset only; no domain adaptation; diagonal = within-dataset subject-grouped 5-fold; off-diagonal = train-all-of-A / test-all-of-B (no target labels)', 'note': 'AI4Pain 2025 and 2026 appear to share one participant cohort; their cross-edition cells are not a test of transfer to new people.', 'matrix': out_matrix}
    (ROOT / 'results' / 'cross_dataset_transfer.json').write_text(json.dumps(out, indent=2))
    print('wrote results/cross_dataset_transfer.json', flush=True)
if __name__ == '__main__':
    main()
