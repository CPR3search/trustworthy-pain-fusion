"""Decodability profile on AI4Pain 2025."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from trustworthy_pain_fusion.data import ai4pain2025 as a25
from trustworthy_pain_fusion.methods.decodability import decodability_profile
ROOT = _PROJECT_ROOT

def targets(y):
    y = np.asarray(y)
    pain = y != 0
    return {'presence (NoPain vs Pain)': ((y != 0).astype(int), np.ones(len(y), bool)), 'intensity (Low vs High)': ((y[pain] == 2).astype(int), pain), 'intensity-3 (NoPain/Low/High)': (y.astype(int), np.ones(len(y), bool))}

def main():
    print('loading AI4Pain 2025 train (nopain_mode=baseline_seg)...', flush=True)
    segs, subj, lab, _ = a25.load_segments(split='train', nopain_mode='baseline_seg')
    segs = a25.normalize_per_subject(segs, subj)
    X = a25.build_features(segs)
    y = np.asarray(lab, int)
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    print(f'X={X.shape}  n_subj={len(set(subj))}  classes={np.bincount(y).tolist()}', flush=True)
    results = {}
    for name, (yt, mask) in targets(y).items():
        p = decodability_profile(X[mask], yt, si[mask])
        results[name] = p
        print(f"  {name:30s} I_V={p['usable_info_bits']:.3f} bits ({100 * p['usable_info_frac_of_H']:.0f}% of H)  acc={p['cv_accuracy']:.3f} chance={1.0 / p['n_classes']:.3f}  null={p['perm_null_usable_info_bits']:.3f}", flush=True)
    out = {'dataset': 'AI4Pain 2025 (intensity)', 'n_segments': int(len(y)), 'n_subjects': int(len(set(subj))), 'n_features': int(X.shape[1]), 'nopain_mode': 'baseline_seg', 'protocol': 'subject-grouped 5-fold, temperature-calibrated V-information (same as methods/decodability.py and biovid_decodability)', 'targets': results}
    dst = ROOT / 'results' / 'ai4pain2025_decodability.json'
    dst.write_text(json.dumps(out, indent=2))
    print('wrote', dst, flush=True)
if __name__ == '__main__':
    main()
