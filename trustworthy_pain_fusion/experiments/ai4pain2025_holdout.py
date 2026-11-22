"""AI4Pain 2025 train-to-validation evaluation."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.data import ai4pain2025 as a25
ROOT = _PROJECT_ROOT

def _fit_eval(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    clf = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, class_weight='balanced', n_jobs=4, verbose=-1)
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return pred

def main():
    sig = a25.SIGNALS
    tr_segs, tr_subj, tr_lab, _ = a25.load_segments(split='train', nopain_mode='baseline_seg')
    va_segs, va_subj, va_lab, _ = a25.load_segments(split='validation', nopain_mode='baseline_seg')
    tr_segs = a25.normalize_per_subject(tr_segs, tr_subj)
    va_segs = a25.normalize_per_subject(va_segs, va_subj)
    Xtr, ytr = (a25.build_features(tr_segs), np.asarray(tr_lab, int))
    Xva, yva = (a25.build_features(va_segs), np.asarray(va_lab, int))
    print(f'train X={Xtr.shape} classes={np.bincount(ytr).tolist()} | val X={Xva.shape} classes={np.bincount(yva).tolist()}', flush=True)
    pred = _fit_eval(Xtr, ytr, Xva, yva)
    acc3 = float((pred == yva).mean())
    det = float(((pred != 0) == (yva != 0)).mean())
    pain = yva != 0
    lowhigh = float((pred[pain] == yva[pain]).mean()) if pain.any() else float('nan')
    out = {'dataset': 'AI4Pain 2025 (intensity)', 'protocol': 'train->validation holdout, LightGBM', 'n_train': int(len(ytr)), 'n_val': int(len(yva)), 'n_val_subjects': int(len(set(va_subj))), 'val_accuracy_3class': round(acc3, 3), 'chance_3class': round(1 / 3, 3), 'val_presence_acc': round(det, 3), 'val_lowhigh_acc_among_truepain': round(lowhigh, 3), 'note': 'leaderboard-comparable val number; cf. AI4Pain 2025 official baseline (~43.3% test, 3-class). Protocol/ split differ (val vs blind test).'}
    dst = ROOT / 'results' / 'ai4pain2025_holdout.json'
    dst.write_text(json.dumps(out, indent=2))
    print(f'VAL 3-class acc {acc3:.3f} | presence {det:.3f} | Low-vs-High {lowhigh:.3f}', flush=True)
    print('wrote', dst, flush=True)
if __name__ == '__main__':
    main()
