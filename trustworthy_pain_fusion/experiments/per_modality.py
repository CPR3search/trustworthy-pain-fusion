"""Per-modality decodability."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from trustworthy_pain_fusion.methods.decodability import decodability_profile
from trustworthy_pain_fusion.data import ai4pain2025 as a25
from trustworthy_pain_fusion.data import biovid
ROOT = _PROJECT_ROOT

def _cols(names, prefix):
    return [i for i, n in enumerate(names) if n.split('_')[0] == prefix]

def _profile_by_modality(X, names, targets, si, modalities):
    out = {}
    for mod in modalities:
        idx = _cols(names, mod)
        Xm = X[:, idx]
        out[mod] = {}
        for tname, (yt, mask) in targets.items():
            p = decodability_profile(Xm[mask], yt, si[mask])
            out[mod][tname] = {'acc': p['cv_accuracy'], 'bacc': p['cv_balanced_accuracy'], 'iv_bits': p['usable_info_bits'], 'iv_unclamped': p['usable_info_unclamped_bits'], 'null': p['perm_null_usable_info_bits'], 'p_value': p['usable_info_p_value'], 'chance': round(1.0 / p['n_classes'], 3)}
            print(f"  {mod:5s} {tname:22s} acc={p['cv_accuracy']:.3f} I_V={p['usable_info_bits']:.3f}", flush=True)
    return out

def run_ai4pain2025():
    segs, subj, lab, _ = a25.load_segments(split='train', nopain_mode='baseline_seg')
    segs = a25.normalize_per_subject(segs, subj)
    X = a25.build_features(segs)
    names = a25.feature_names()
    y = np.asarray(lab, int)
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    pain = y != 0
    targets = {'presence (NoPain vs Pain)': ((y != 0).astype(int), np.ones(len(y), bool)), 'intensity (Low vs High)': ((y[pain] == 2).astype(int), pain)}
    print('AI4Pain 2025 per-modality:', flush=True)
    return _profile_by_modality(X, names, targets, si, ['Bvp', 'Eda', 'Resp', 'SpO2'])

def run_biovid():
    segs, subj, lab, _ = biovid.load_segments()
    segs = biovid.normalize_per_subject(segs, subj)
    X = biovid.build_features(segs)
    names = biovid.feature_names()
    y = np.asarray(lab, int)
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    m = (y == 0) | (y == 4)
    targets = {'presence (BL1 vs PA4)': ((y[m] == 4).astype(int), m), 'intensity (BL1..PA4)': (y.astype(int), np.ones(len(y), bool))}
    print('BioVid per-modality:', flush=True)
    out = {}
    for mod, pref_match in [('gsr', lambda n: n.startswith('gsr_')), ('ecg', lambda n: n.startswith('ecg_')), ('emg_trapezius', lambda n: n.startswith('emg_trapezius_'))]:
        idx = [i for i, n in enumerate(names) if pref_match(n)]
        Xm = X[:, idx]
        out[mod] = {}
        for tname, (yt, mask) in targets.items():
            p = decodability_profile(Xm[mask], yt, si[mask])
            out[mod][tname] = {'acc': p['cv_accuracy'], 'bacc': p['cv_balanced_accuracy'], 'iv_bits': p['usable_info_bits'], 'iv_unclamped': p['usable_info_unclamped_bits'], 'null': p['perm_null_usable_info_bits'], 'p_value': p['usable_info_p_value'], 'chance': round(1.0 / p['n_classes'], 3)}
            print(f"  {mod:13s} {tname:22s} acc={p['cv_accuracy']:.3f} I_V={p['usable_info_bits']:.3f}", flush=True)
    return out

def main():
    out = {'ai4pain2025': run_ai4pain2025(), 'biovid': run_biovid(), 'protocol': 'subject-grouped 5-fold, temperature-calibrated V-information, per single modality'}
    (ROOT / 'results' / 'per_modality_decodability.json').write_text(json.dumps(out, indent=2))
    print('wrote results/per_modality_decodability.json', flush=True)
if __name__ == '__main__':
    main()
