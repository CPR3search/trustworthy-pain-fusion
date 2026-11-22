"""AI4Pain 2026 loader, per-subject normalization and features."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats
FS = 100
SIGNALS = ['Bvp', 'Eda', 'Resp', 'SpO2']
CLASS_MAP = {'Baseline': 0, 'ARM': 1, 'HAND': 2}
CLASS_NAMES = ['NoPain', 'PainArm', 'PainHand']

def parse_header(col: str):
    parts = col.split('_')
    subj = parts[0]
    if len(parts) >= 3 and parts[1] in CLASS_MAP:
        return (subj, CLASS_MAP[parts[1]])
    return (subj, None)

def load_segments(data_dir, split, signals=SIGNALS):
    base = Path(data_dir) / split
    subj_files = sorted((base / signals[0]).glob('*.csv'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not subj_files:
        raise FileNotFoundError(f'no CSVs under {(base / signals[0]).resolve()}')
    segments, subjects, labels, seg_ids = ([], [], [], [])
    for f in subj_files:
        subj = f.stem
        dfs = {s: pd.read_csv(Path(data_dir) / split / s / f.name) for s in signals}
        cols = list(dfs[signals[0]].columns)
        for s in signals[1:]:
            if set(dfs[s].columns) != set(cols):
                raise ValueError(f'{subj}: header mismatch between {signals[0]} and {s}')
        for c in cols:
            _, lab = parse_header(c)
            seg = {s: dfs[s][c].to_numpy(dtype=float) for s in signals}
            segments.append(seg)
            subjects.append(subj)
            labels.append(lab)
            seg_ids.append(c)
    labels = None if any((l is None for l in labels)) else labels
    return (segments, subjects, labels, seg_ids)

def normalize_per_subject(segments, subjects, signals=SIGNALS):
    subjects = np.asarray(subjects)
    out = [dict(seg) for seg in segments]
    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        for s in signals:
            allvals = np.concatenate([segments[i][s] for i in idx])
            mu, sd = (allvals.mean(), allvals.std() + 1e-08)
            for i in idx:
                out[i][s] = (segments[i][s] - mu) / sd
    return out

def _generic(x):
    x = np.asarray(x, float)
    diff = np.diff(x) if len(x) > 1 else np.array([0.0])
    slope = np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0.0
    q75, q25 = np.percentile(x, [75, 25])
    return [x.mean(), x.std(), x.min(), x.max(), x.max() - x.min(), np.median(x), q75 - q25, slope, np.mean(np.abs(diff)), np.sqrt(np.mean(x ** 2)), float(sp_stats.skew(x)), float(sp_stats.kurtosis(x))]
_GENERIC_NAMES = ['mean', 'std', 'min', 'max', 'range', 'median', 'iqr', 'slope', 'mad', 'rms', 'skew', 'kurt']

def _hjorth(x):
    x = np.asarray(x, float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    var0 = x.var() + 1e-12
    var1 = dx.var() + 1e-12
    var2 = ddx.var() + 1e-12
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return [var0, mobility, complexity]
_HJORTH_NAMES = ['hj_activity', 'hj_mobility', 'hj_complexity']

def _spectral(x, fs=FS):
    x = np.asarray(x, float)
    nperseg = min(len(x), 256)
    f, p = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    p = p + 1e-12
    tot = p.sum()
    pn = p / tot
    spec_entropy = float(-(pn * np.log(pn)).sum() / np.log(len(pn)))
    peak_f = float(f[np.argmax(p)])
    low = p[(f >= 0.04) & (f < 0.5)].sum()
    high = p[(f >= 0.5) & (f < 5.0)].sum()
    ratio = float(low / (high + 1e-12))
    return [float(tot), peak_f, spec_entropy, ratio]
_SPECTRAL_NAMES = ['spec_totpow', 'spec_peakf', 'spec_entropy', 'spec_lh_ratio']

def _dynamics(x):
    x = np.asarray(x, float)
    n = len(x)
    half = n // 2
    first, second = (x[:half].mean(), x[half:].mean())
    rng = x.max() - x.min() + 1e-12
    early = np.sum(x[:half] ** 2)
    late = np.sum(x[half:] ** 2)
    ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if n > 2 else 0.0
    return [first, second, second - first, float(np.argmax(x)) / n, float(np.argmin(x)) / n, float(early / (late + 1e-12)), ac1, float((x.max() - x[0]) / rng)]
_DYNAMICS_NAMES = ['half1_mean', 'half2_mean', 'half_delta', 't_max', 't_min', 'early_late_energy', 'autocorr1', 'peak_rise_frac']

def _eda(x, fs=FS):
    x = np.asarray(x, float)
    win = max(3, int(fs * 2))
    kernel = np.ones(win) / win
    tonic = np.convolve(x, kernel, mode='same')
    phasic = x - tonic
    pamp = phasic.std() + 1e-08
    peaks, props = sp_signal.find_peaks(phasic, height=pamp, distance=int(fs * 1))
    n_scr = len(peaks)
    scr_amp = float(phasic[peaks].mean()) if n_scr else 0.0
    scr_max = float(phasic[peaks].max()) if n_scr else 0.0
    scr_sum = float(phasic[peaks].sum()) if n_scr else 0.0
    phasic_auc = float(np.clip(phasic, 0, None).sum() / fs)
    tonic_slope = np.polyfit(np.arange(len(tonic)), tonic, 1)[0]
    return [tonic.mean(), tonic[-1] - tonic[0], tonic_slope, phasic.std(), float(n_scr), float(n_scr) / (len(x) / fs), scr_amp, scr_max, scr_sum, phasic_auc]
_EDA_NAMES = ['tonic_mean', 'tonic_rise', 'tonic_slope', 'phasic_std', 'n_scr', 'scr_rate', 'scr_amp', 'scr_max', 'scr_sum', 'phasic_auc']

def _bvp(x, fs=FS):
    x = np.asarray(x, float)
    peaks, _ = sp_signal.find_peaks(x, distance=int(fs * 0.4))
    if len(peaks) >= 3:
        ibi = np.diff(peaks) / fs
        dibi = np.diff(ibi)
        hr = 60.0 / ibi.mean()
        sdnn = ibi.std()
        rmssd = np.sqrt(np.mean(dibi ** 2)) if len(dibi) else 0.0
        pnn = float(np.mean(np.abs(dibi) > 0.05)) if len(dibi) else 0.0
        ibi_range = float(ibi.max() - ibi.min())
        amp = float(x[peaks].mean())
        amp_std = float(x[peaks].std())
    else:
        hr = sdnn = rmssd = pnn = ibi_range = amp = amp_std = 0.0
    return [hr, sdnn, rmssd, pnn, ibi_range, float(len(peaks)), amp, amp_std]
_BVP_NAMES = ['hr', 'sdnn', 'rmssd', 'pnn50', 'ibi_range', 'n_peaks', 'pulse_amp', 'pulse_amp_std']

def _resp(x, fs=FS):
    x = np.asarray(x, float)
    peaks, _ = sp_signal.find_peaks(x, distance=int(fs * 1.0))
    troughs, _ = sp_signal.find_peaks(-x, distance=int(fs * 1.0))
    if len(peaks) >= 2:
        bi = np.diff(peaks) / fs
        rate = 60.0 / bi.mean()
        bivar = bi.std()
    else:
        rate = bivar = 0.0
    depth = float(x[peaks].mean() - x[troughs].mean()) if len(peaks) and len(troughs) else 0.0
    return [rate, bivar, float(len(peaks)), depth]
_RESP_NAMES = ['resp_rate', 'breath_var', 'n_breaths', 'resp_depth']
_SPECIALIZED = {'Eda': (_eda, _EDA_NAMES), 'Bvp': (_bvp, _BVP_NAMES), 'Resp': (_resp, _RESP_NAMES)}
_SHARED_NAMES = _GENERIC_NAMES + _HJORTH_NAMES + _SPECTRAL_NAMES + _DYNAMICS_NAMES

def feature_names(signals=SIGNALS, cross=True):
    names = []
    for s in signals:
        names += [f'{s}_{n}' for n in _SHARED_NAMES]
        if s in _SPECIALIZED:
            names += [f'{s}_{n}' for n in _SPECIALIZED[s][1]]
    if cross:
        names += _cross_names(signals)
    return names

def _cross(seg, signals=SIGNALS):
    feats = []
    L = min((len(seg[s]) for s in signals))
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            a, b = (seg[signals[i]][:L], seg[signals[j]][:L])
            if a.std() < 1e-09 or b.std() < 1e-09:
                feats.append(0.0)
            else:
                feats.append(float(np.corrcoef(a, b)[0, 1]))
    return feats

def _cross_names(signals=SIGNALS):
    return [f'corr_{signals[i]}_{signals[j]}' for i in range(len(signals)) for j in range(i + 1, len(signals))]

def feature_names_cross(signals=SIGNALS):
    return _cross_names(signals)

def segment_features(seg, signals=SIGNALS, cross=True):
    feats = []
    for s in signals:
        x = seg[s]
        feats += _generic(x)
        feats += _hjorth(x)
        feats += _spectral(x)
        feats += _dynamics(x)
        if s in _SPECIALIZED:
            fn, _ = _SPECIALIZED[s]
            feats += fn(x)
    if cross:
        feats += _cross(seg, signals)
    return np.asarray(feats, float)

def build_features(segments, signals=SIGNALS):
    X = np.vstack([segment_features(seg, signals) for seg in segments])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class AI4Pain2026Dataset:
    """AI4Pain 2026: segments, per-subject normalization and features."""
    signals = SIGNALS
    load_segments = staticmethod(load_segments)
    normalize_per_subject = staticmethod(normalize_per_subject)
    build_features = staticmethod(build_features)
