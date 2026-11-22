"""Decodability profile on BioVid Part A."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from trustworthy_pain_fusion.data import biovid
from trustworthy_pain_fusion.methods.decodability import decodability_profile
ROOT = _PROJECT_ROOT

def main():
    print('loading BioVid Part A (8700 trials)...', flush=True)
    segs, subj, lab, _ = biovid.load_segments()
    segs = biovid.normalize_per_subject(segs, subj)
    X = biovid.build_features(segs)
    y = np.asarray(lab, int)
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    print(f'X={X.shape}  n_subj={len(set(subj))}  classes={np.bincount(y).tolist()}', flush=True)
    results = {}
    for name, (yt, mask) in biovid.biovid_targets(y).items():
        p = decodability_profile(X[mask], yt, si[mask])
        results[name] = p
        print(f"  {name:24s} I_V={p['usable_info_bits']:.3f} bits ({100 * p['usable_info_frac_of_H']:.0f}% of H)  acc={p['cv_accuracy']:.3f}  chance={1.0 / p['n_classes']:.3f}  null={p['perm_null_usable_info_bits']:.3f}", flush=True)
    out = {'dataset': 'BioVid Part A (biosignals_filtered)', 'n_trials': int(len(y)), 'n_subjects': int(len(set(subj))), 'n_features': int(X.shape[1]), 'channels': biovid.CHANNELS, 'representation': 'hand-crafted (shared descriptors, fs=512)', 'targets': results}
    dst = ROOT / 'results' / 'biovid_decodability.json'
    dst.write_text(json.dumps(out, indent=2))
    print('wrote', dst, flush=True)
if __name__ == '__main__':
    main()
