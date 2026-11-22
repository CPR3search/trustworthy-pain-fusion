"""Within-subject condition effects with FDR control."""
import json
from pathlib import Path
import numpy as np
from scipy import stats
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain

def _bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    r = p[o] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = q
    return np.clip(out, 0, 1)

def _paired_ate(diffs):
    n = diffs.shape[0]
    mean = np.nanmean(diffs, 0)
    sd = np.nanstd(diffs, 0, ddof=1)
    dz = mean / (sd + 1e-12)
    t, p = stats.ttest_1samp(diffs, 0.0, axis=0, nan_policy='omit')
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci_lo, ci_hi = (mean - tcrit * se, mean + tcrit * se)
    q = _bh(np.nan_to_num(p, nan=1.0))
    return (mean, dz, p, q, ci_lo, ci_hi)

def main():
    segs, subj, lab, _ = ai4pain.load_segments('data', 'train', ai4pain.SIGNALS)
    segs = ai4pain.normalize_per_subject(segs, subj, ai4pain.SIGNALS)
    names = ai4pain.feature_names()
    X = ai4pain.build_features(segs, ai4pain.SIGNALS)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    y = np.asarray(lab, int)
    subj = np.asarray(subj)
    subs = np.unique(subj)
    pain_d, site_d = ([], [])
    for s in subs:
        m = subj == s
        nopain = X[m & (y == 0)].mean(0)
        pain = X[m & (y != 0)].mean(0)
        arm = X[m & (y == 1)].mean(0)
        hand = X[m & (y == 2)].mean(0)
        pain_d.append(pain - nopain)
        site_d.append(hand - arm)
    pain_d, site_d = (np.array(pain_d), np.array(site_d))
    rep = {'n_subjects': int(len(subs)), 'n_features': len(names), 'scope': 'experimental (healthy adults, TENS-evoked acute pain); NOT clinical/diagnostic evidence'}
    for label, d in [('pain_ATE', pain_d), ('site_ATE', site_d)]:
        mean, dz, p, q, lo, hi = _paired_ate(d)
        sig = int((q < 0.05).sum())
        order = np.argsort(np.abs(dz))[::-1][:10]
        rep[label] = {'n_sig_fdr05': sig, 'max_abs_dz': round(float(np.abs(dz).max()), 3), 'top': [{'feature': names[j], 'dz': round(float(dz[j]), 3), 'ate_sd': round(float(mean[j]), 3), 'ci': [round(float(lo[j]), 3), round(float(hi[j]), 3)], 'q': round(float(q[j]), 4)} for j in order]}
        print(f'== {label}: {sig}/{len(names)} features significant (FDR<0.05), max |d_z|={np.abs(dz).max():.2f} ==')
        for j in order[:6]:
            print(f'  {names[j]:22s} d_z={dz[j]:+.2f}  ATE={mean[j]:+.2f} SD [{lo[j]:+.2f},{hi[j]:+.2f}]  q={q[j]:.3f}')
        print()
    Path('results/causal_ate.json').write_text(json.dumps(rep, indent=2))
    print('wrote results/causal_ate.json')
if __name__ == '__main__':
    main()
