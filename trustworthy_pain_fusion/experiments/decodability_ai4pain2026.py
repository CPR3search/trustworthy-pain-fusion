"""Decodability profile on AI4Pain 2026."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
from trustworthy_pain_fusion.methods.decodability import decodability_profile, ai4pain_targets
ROOT = _PROJECT_ROOT

def main():
    print('loading AI4Pain 2026 train...', flush=True)
    segs, subj, lab, _ = ai4pain.load_segments('data', 'train', ai4pain.SIGNALS)
    segs = ai4pain.normalize_per_subject(segs, subj, ai4pain.SIGNALS)
    X = ai4pain.build_features(segs, ai4pain.SIGNALS)
    y = np.asarray(lab, int)
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    print(f'X={X.shape}  n_subj={len(set(subj))}  classes={np.bincount(y).tolist()}', flush=True)
    rename = {'presence (NoPain vs Pain)': 'presence (NoPain vs Pain)', 'location (Arm vs Hand)': 'location (Arm vs Hand)'}
    results = {}
    for name, (yt, mask) in ai4pain_targets(y).items():
        p = decodability_profile(X[mask], yt, si[mask])
        results[rename[name]] = p
        print(f"  {name:30s} I_V={p['usable_info_bits']:.3f} bits ({100 * p['usable_info_frac_of_H']:.0f}% of H)  acc={p['cv_accuracy']:.3f} chance={1.0 / p['n_classes']:.3f}  null={p['perm_null_usable_info_bits']:.3f}", flush=True)
    out = {'dataset': 'AI4Pain 2026 (location)', 'n_segments': int(len(y)), 'n_subjects': int(len(set(subj))), 'n_features': int(X.shape[1]), 'protocol': 'subject-grouped 5-fold, temperature-calibrated V-information (same estimator as ai4pain2025_decodability and biovid_decodability)', 'note': 'Calibrated recompute; supersedes the uncalibrated conference result for the cross-dataset table. Location = 0.000 under both calibrated and uncalibrated estimators.', 'targets': results}
    dst = ROOT / 'results' / 'ai4pain2026_decodability.json'
    dst.write_text(json.dumps(out, indent=2))
    print('wrote', dst, flush=True)
if __name__ == '__main__':
    main()
