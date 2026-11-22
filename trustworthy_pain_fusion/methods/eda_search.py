"""EDA representation search for stimulation site."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
import math
import sys
from pathlib import Path
import numpy as np
import pywt
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
from trustworthy_pain_fusion.methods.splits import loso_splits, grouped_kfold
FS = 100
EPS = 1e-12
CACHE = _PROJECT_ROOT / 'cache' / 'eda_featbank.npz'

def f_stat(x):
    ax = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2)) + EPS
    p = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
    v = [x.mean(), x.std(), x.var(), float(sp_stats.skew(x)), float(sp_stats.kurtosis(x)), x.min(), x.max(), x.max() - x.min(), rms, np.mean(ax), np.median(ax), float(np.mean(np.abs(x - x.mean()))), p[5] - p[1], *p, x.std() / (abs(x.mean()) + EPS)]
    n = ['mean', 'std', 'var', 'skew', 'kurt', 'min', 'max', 'range', 'rms', 'absmean', 'absmedian', 'mad', 'iqr8010', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'cv']
    return (v, [f'stat_{k}' for k in n])

def f_shape(x):
    ax = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2)) + EPS
    am = ax.mean() + EPS
    zc = int(np.sum(np.diff(np.signbit(x)) != 0))
    mc = int(np.sum(np.diff(np.signbit(x - x.mean())) != 0))
    v = [ax.max() / rms, rms / am, ax.max() / am, ax.max() / (np.mean(np.sqrt(ax)) ** 2 + EPS), zc, mc]
    return (v, [f'shape_{k}' for k in ['crest', 'shapef', 'impulse', 'margin', 'zcross', 'mcross']])

def f_hjorth(x):
    dx, ddx = (np.diff(x), np.diff(np.diff(x)))
    v0, v1, v2 = (x.var() + EPS, dx.var() + EPS, ddx.var() + EPS)
    mob = np.sqrt(v1 / v0)
    v = [v0, mob, np.sqrt(v2 / v1) / (mob + EPS)]
    return (v, ['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'])

def f_deriv(x):
    d1, d2 = (np.diff(x), np.diff(np.diff(x)))
    v = [np.mean(np.abs(d1)), d1.std(), np.max(np.abs(d1)), np.sum(d1 ** 2), np.mean(np.abs(d2)), d2.std(), np.max(np.abs(d2)), float(np.polyfit(np.arange(len(x)), x, 1)[0])]
    n = ['d1absmean', 'd1std', 'd1max', 'd1energy', 'd2absmean', 'd2std', 'd2max', 'slope']
    return (v, [f'deriv_{k}' for k in n])

def f_dynamics(x):
    n = len(x)
    h = n // 2
    e_early, e_late = (np.sum(x[:h] ** 2), np.sum(x[h:] ** 2))
    ac = [float(np.corrcoef(x[:-k], x[k:])[0, 1]) if n > k + 2 else 0.0 for k in (1, 5, 10, 25, 50)]
    v = [e_early, e_late, e_late - e_early, e_late / (e_early + EPS), x[:h].mean(), x[h:].mean(), x[h:].mean() - x[:h].mean(), np.trapezoid(x), *ac]
    n_ = ['e_early', 'e_late', 'e_diff', 'e_ratio', 'm_early', 'm_late', 'm_diff', 'auc', 'ac1', 'ac5', 'ac10', 'ac25', 'ac50']
    return (v, [f'dyn_{k}' for k in n_])

def _tonic_phasic(x, fs=FS):
    k = int(round(2.0 * fs)) | 1
    tonic = sp_signal.medfilt(x, kernel_size=min(k, len(x) - (1 - len(x) % 2)))
    return (tonic, x - tonic)

def f_scr(x, fs=FS):
    tonic, phasic = _tonic_phasic(x, fs)
    pk, props = sp_signal.find_peaks(phasic, prominence=0.05 * (phasic.std() + EPS))
    amps = props.get('prominences', np.array([]))
    ipi = np.diff(pk) if len(pk) > 1 else np.array([0.0])
    v = [len(pk), amps.mean() if len(amps) else 0.0, amps.max() if len(amps) else 0.0, amps.std() if len(amps) else 0.0, ipi.mean(), ipi.std(), float(pk[0]) if len(pk) else 0.0, float(pk[np.argmax(amps)]) if len(amps) else 0.0, tonic.mean(), float(np.polyfit(np.arange(len(tonic)), tonic, 1)[0]), tonic.max() - tonic.min(), np.sum(phasic ** 2), np.trapezoid(np.abs(phasic)), phasic.std(), phasic.max()]
    n = ['npeaks', 'amp_mean', 'amp_max', 'amp_std', 'ipi_mean', 'ipi_std', 't_first', 't_max', 'tonic_mean', 'tonic_slope', 'tonic_range', 'phasic_energy', 'phasic_auc', 'phasic_std', 'phasic_max']
    return (v, [f'scr_{k}' for k in n])

def f_freq(x, fs=FS):
    fr, pxx = sp_signal.welch(x, fs=fs, nperseg=len(x))
    pxx = pxx + EPS
    tot = pxx.sum()
    bands = [(0, 0.045), (0.045, 0.15), (0.15, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0)]
    bp = [pxx[(fr >= lo) & (fr < hi)].sum() for lo, hi in bands]
    bpn = [b / tot for b in bp]
    pn = pxx / tot
    centroid = np.sum(fr * pn)
    spread = np.sqrt(np.sum((fr - centroid) ** 2 * pn))
    ent = -np.sum(pn * np.log(pn)) / np.log(len(pn))
    flat = np.exp(np.mean(np.log(pxx))) / np.mean(pxx)
    cum = np.cumsum(pxx) / tot
    roll85 = fr[np.searchsorted(cum, 0.85)] if cum[-1] >= 0.85 else fr[-1]
    roll95 = fr[np.searchsorted(cum, 0.95)] if cum[-1] >= 0.95 else fr[-1]
    medf = fr[np.searchsorted(cum, 0.5)]
    peakf = fr[np.argmax(pxx)]
    edasymp = pxx[(fr >= 0.045) & (fr < 0.25)].sum() / tot
    v = [tot, *bp, *bpn, centroid, spread, ent, flat, roll85, roll95, medf, peakf, edasymp, bp[1] / (bp[0] + EPS), bp[2] / (bp[1] + EPS)]
    n = ['totpow'] + [f'bp{i}' for i in range(6)] + [f'bpn{i}' for i in range(6)] + ['centroid', 'spread', 'specent', 'flatness', 'roll85', 'roll95', 'medfreq', 'peakfreq', 'edasymp', 'ratio10', 'ratio21']
    return (v, [f'freq_{k}' for k in n])

def f_wavelet(x, wave='db4', level=5):
    lv = min(level, pywt.dwt_max_level(len(x), pywt.Wavelet(wave).dec_len))
    coeffs = pywt.wavedec(x, wave, level=max(lv, 1))
    energies = np.array([np.sum(c ** 2) for c in coeffs]) + EPS
    tot = energies.sum()
    v, n = ([], [])
    for i, c in enumerate(coeffs):
        pe = energies[i] / tot
        p = c ** 2 / (np.sum(c ** 2) + EPS)
        v += [pe, -np.sum(p * np.log(p + EPS)), c.std(), float(np.mean(np.abs(c)))]
        n += [f'wl{i}_relE', f'wl{i}_ent', f'wl{i}_std', f'wl{i}_absmean']
    target = 7 * 4
    if len(v) < target:
        v += [0.0] * (target - len(v))
        n += [f'wl_pad{i}' for i in range(target - len(n))]
    return (v[:target], n[:target])

def _decimate(x, q=4):
    return sp_signal.decimate(x, q, ftype='fir') if len(x) > 30 * q else x

def _sampen(x, m=2, r=0.2):
    x = (x - x.mean()) / (x.std() + EPS)
    n = len(x)
    r *= 1.0

    def phi(mm):
        X = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        C = 0
        for i in range(len(X)):
            d = np.max(np.abs(X - X[i]), axis=1)
            C += np.sum(d <= r) - 1
        return C
    a, b = (phi(m + 1), phi(m))
    return float(-np.log((a + EPS) / (b + EPS)))

def _perm_entropy(x, order=3, delay=1):
    n = len(x)
    patterns = {}
    for i in range(n - delay * (order - 1)):
        vec = tuple(np.argsort(x[i:i + delay * order:delay]))
        patterns[vec] = patterns.get(vec, 0) + 1
    c = np.array(list(patterns.values()), float)
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)) / np.log(math.factorial(order)))

def _higuchi(x, kmax=8):
    n = len(x)
    L = []
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            ll = np.sum(np.abs(np.diff(x[idx]))) * (n - 1) / (len(idx) * k)
            Lk.append(ll)
        if Lk:
            L.append(np.log(np.mean(Lk) + EPS))
    lnk = np.log(1.0 / np.arange(1, len(L) + 1))
    return float(np.polyfit(lnk, L, 1)[0]) if len(L) > 1 else 0.0

def _dfa(x, scales=(4, 8, 16, 32)):
    y = np.cumsum(x - x.mean())
    F = []
    for s in scales:
        if len(y) < 2 * s:
            continue
        segs = len(y) // s
        rms = []
        for j in range(segs):
            seg = y[j * s:(j + 1) * s]
            t = np.arange(s)
            fit = np.polyval(np.polyfit(t, seg, 1), t)
            rms.append(np.sqrt(np.mean((seg - fit) ** 2)))
        F.append((s, np.mean(rms) + EPS))
    if len(F) < 2:
        return 0.0
    s_, f_ = (np.array([a[0] for a in F]), np.array([a[1] for a in F]))
    return float(np.polyfit(np.log(s_), np.log(f_), 1)[0])

def f_nonlinear(x):
    xd = _decimate(x, 4)
    tp = int(np.sum(np.diff(x[:-1]) * np.diff(x[1:]) < 0))
    v = [_sampen(xd), _perm_entropy(xd, 3), _perm_entropy(xd, 4), _higuchi(xd), _dfa(xd), tp / len(x)]
    return (v, [f'nl_{k}' for k in ['sampen', 'pe3', 'pe4', 'higuchi', 'dfa', 'tpr']])

def f_cepstral(x, ncoef=13):
    spec = np.abs(np.fft.fft(x)) + EPS
    ceps = np.real(np.fft.ifft(np.log(spec)))[:ncoef]
    return (list(ceps), [f'ceps_{i}' for i in range(ncoef)])
FAMILIES = [f_stat, f_shape, f_hjorth, f_deriv, f_dynamics, f_scr, f_freq, f_wavelet, f_nonlinear, f_cepstral]

def extract(x):
    vals, names = ([], [])
    for fn in FAMILIES:
        v, n = fn(x)
        vals += list(v)
        names += n
    return (np.array(vals, float), names)

def _fam_of(name):
    if name.startswith('bp'):
        return name.split('_')[0]
    p = name.split('_')[0]
    return 'wl' if p.startswith('wl') else p

def build_features():
    from trustworthy_pain_fusion.methods.eda_patterns import bp_bank
    segs, subj, lab, _ = ai4pain.load_segments('data', 'train', ['Eda'])
    segs = ai4pain.normalize_per_subject(segs, subj, ['Eda'])
    X, names = ([], None)
    for s in segs:
        ve, ne = extract(s['Eda'])
        vb, nb = bp_bank(s['Eda'])
        X.append(np.concatenate([ve, vb]))
        names = ne + nb
    X = np.nan_to_num(np.vstack(X), nan=0.0, posinf=0.0, neginf=0.0)
    subj = np.asarray(subj)
    lab = np.asarray(lab, int)
    _, si = np.unique(subj, return_inverse=True)
    fam = np.array([_fam_of(n) for n in names])
    group = np.array(['bp' if n.startswith('bp') else 'eda' for n in names])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, lab=lab, si=si, names=np.array(names), fam=fam, group=group)
    print(f'cached {X.shape} features ({len(names)}) -> {CACHE.name}')
    print('groups:', {g: int((group == g).sum()) for g in dict.fromkeys(group)})
    print('families:', {f: int((fam == f).sum()) for f in dict.fromkeys(fam)})
    return (X, lab, si, names, fam, group)

def _pipe(kind, k=None):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    clf = {'logreg': LogisticRegression(max_iter=2000, class_weight='balanced'), 'lda': LinearDiscriminantAnalysis(), 'svm': SVC(C=10, gamma='scale', class_weight='balanced'), 'rf': RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight='balanced', random_state=0)}[kind]
    steps = [('sc', StandardScaler())]
    if k is not None:
        steps.append(('sel', SelectKBest(mutual_info_classif, k=k)))
    steps.append(('clf', clf))
    return Pipeline(steps)

def _lgbm(k=None):
    from lightgbm import LGBMClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    steps = []
    if k is not None:
        steps.append(('sel', SelectKBest(mutual_info_classif, k=k)))
    steps.append(('clf', LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, class_weight='balanced', n_jobs=1, verbose=-1)))
    return Pipeline(steps)

def _loso_bal(X, y, si, make):
    from sklearn.metrics import balanced_accuracy_score
    accs = []
    for tr, te in loso_splits(si):
        m = make().fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return (float(np.mean(accs)), float(np.std(accs)))

def _task(lab, task):
    if task == 'location':
        m = np.isin(lab, [1, 2])
        return ((lab[m] == 2).astype(int), m)
    return ((lab > 0).astype(int), np.ones(len(lab), bool))

def evaluate():
    d = np.load(CACHE, allow_pickle=True)
    X, lab, si, names, fam, group = (d['X'], d['lab'], d['si'], d['names'], d['fam'], d['group'])
    for task in ('location', 'detection'):
        y, mask = _task(lab, task)
        Xt, st = (X[mask], si[mask])
        chance = 50.0
        print(f"\n{'=' * 66}\n{task.upper()}  (Arm/Hand)" if task == 'location' else f"\n{'=' * 66}\n{task.upper()}  (NoPain/Pain)")
        print(f'n={len(y)}  balanced-acc LOSO (chance {chance:.0f}%)\n')
        print('  -- full bank (137 EDA + BP), by classifier --')
        for kind, mk in [('logreg', lambda: _pipe('logreg')), ('lda', lambda: _pipe('lda', k=30)), ('svm', lambda: _pipe('svm', k=40)), ('rf', lambda: _pipe('rf')), ('lgbm', lambda: _lgbm())]:
            m, s = _loso_bal(Xt, y, st, mk)
            print(f'     {kind:8s} {m * 100:5.1f} +/- {s * 100:4.1f}')
        for k in (10, 20, 40, 80):
            m, s = _loso_bal(Xt, y, st, lambda k=k: _lgbm(k=k))
            print(f'     lgbm+MI{k:<3d} {m * 100:5.1f} +/- {s * 100:4.1f}')
        print('  -- by feature group / family (lgbm) --')
        for g in ('eda', 'bp'):
            gi = np.where(group == g)[0]
            m, _ = _loso_bal(Xt[:, gi], y, st, lambda: _lgbm())
            print(f'     [{g}] ({len(gi)}) {m * 100:5.1f}')
        best_fam = None
        for f in dict.fromkeys(fam):
            fi = np.where(fam == f)[0]
            m, _ = _loso_bal(Xt[:, fi], y, st, lambda: _lgbm())
            tag = '  <-- best' if best_fam is None or m > best_fam[1] else ''
            if best_fam is None or m > best_fam[1]:
                best_fam = (f, m)
            print(f'     {f:10s} ({len(fi):3d}) {m * 100:5.1f}{tag}')
FAM_DISPLAY = {'stat': 'Statistics', 'shape': 'Shape', 'hjorth': 'Hjorth', 'deriv': 'Derivative', 'dyn': 'Dynamics', 'scr': 'SCR/tonic-phasic', 'freq': 'Spectral', 'wl': 'Wavelet', 'nl': 'Nonlinear', 'ceps': 'Cepstral', 'bpLBP': 'LBP', 'bpLBPR2': 'LBP-R2', 'bpCSLBP': 'CS-LBP', 'bpLTP': 'LTP', 'bpGLBP': 'Gradient-LBP'}

def families():
    d = np.load(CACHE, allow_pickle=True)
    X, lab, si, fam, group = (d['X'], d['lab'], d['si'], d['fam'], d['group'])
    out = {'chance': 50.0, 'families': {}, 'groups': {}}
    for f in dict.fromkeys(fam):
        fi = np.where(fam == f)[0]
        row = {'n': int(len(fi)), 'group': 'bp' if str(f).startswith('bp') else 'eda', 'display': FAM_DISPLAY.get(str(f), str(f))}
        for task in ('detection', 'location'):
            y, m = _task(lab, task)
            row[task] = round(_loso_bal(X[m][:, fi], y, si[m], lambda: _lgbm())[0] * 100, 2)
        out['families'][str(f)] = row
    for g in ('eda', 'bp'):
        gi = np.where(group == g)[0]
        row = {'n': int(len(gi))}
        for task in ('detection', 'location'):
            y, m = _task(lab, task)
            row[task] = round(_loso_bal(X[m][:, gi], y, si[m], lambda: _lgbm())[0] * 100, 2)
        out['groups'][g] = row
    Path('results/eda_localise_families.json').write_text(json.dumps(out, indent=2))
    print('wrote results/eda_localise_families.json')
    for f, r in out['families'].items():
        print(f"  {r['display']:16s} det {r['detection']:5.1f}  loc {r['location']:5.1f}")
if __name__ == '__main__':
    sys.argv = sys.argv[:1] + sys.argv[1:2]
    if 'features' in sys.argv:
        build_features()
    elif 'evaluate' in sys.argv:
        evaluate()
    elif 'families' in sys.argv:
        families()
    else:
        raise SystemExit('usage: eda-search {features,evaluate,families}')
