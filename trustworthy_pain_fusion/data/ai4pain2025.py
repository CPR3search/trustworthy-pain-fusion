"""AI4Pain 2025 loader, per-subject normalization and features."""
from pathlib import Path
import numpy as np
import pandas as pd
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
FS = ai4pain.FS
SIGNALS = ai4pain.SIGNALS
WIN = 1022
CLASS_MAP = {'Baseline': 0, 'LOW': 1, 'HIGH': 2}
CLASS_NAMES = ['NoPain', 'Low', 'High']
DATA_DIR = 'data/ai4pain2025'

def parse_header(col):
    parts = col.split('_')
    if len(parts) >= 3 and parts[1] in ('Baseline', 'LOW', 'HIGH', 'REST'):
        return (parts[0], parts[1])
    return (parts[0], None)

def _col_signal(df, col):
    x = df[col].to_numpy(dtype=float)
    return x[~np.isnan(x)]

def _windows(x, win=WIN):
    n = len(x) // win
    return [x[i * win:(i + 1) * win] for i in range(n)] if n else [x] if len(x) else []

def load_segments(data_dir=DATA_DIR, split='train', signals=SIGNALS, nopain_mode='baseline_seg'):
    if nopain_mode not in ('baseline_seg', 'baseline_rest', 'baseline_whole'):
        raise ValueError(f'bad nopain_mode {nopain_mode!r}')
    base = Path(data_dir) / split
    subj_files = sorted((base / signals[0]).glob('*.csv'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not subj_files:
        raise FileNotFoundError(f'no CSVs under {(base / signals[0]).resolve()}')
    segments, subjects, labels, seg_ids = ([], [], [], [])
    for f in subj_files:
        subj = f.stem
        dfs = {s: pd.read_csv(base / s / f.name) for s in signals}
        cols = list(dfs[signals[0]].columns)
        for s in signals[1:]:
            if set(dfs[s].columns) != set(cols):
                raise ValueError(f'{subj}: header mismatch {signals[0]} vs {s}')
        for c in cols:
            _, tok = parse_header(c)
            if tok in ('LOW', 'HIGH'):
                wins_per_signal = {s: [_col_signal(dfs[s], c)] for s in signals}
                lab = CLASS_MAP[tok]
            elif tok == 'Baseline':
                if nopain_mode == 'baseline_whole':
                    wins_per_signal = {s: [_col_signal(dfs[s], c)] for s in signals}
                else:
                    wins_per_signal = {s: _windows(_col_signal(dfs[s], c)) for s in signals}
                lab = 0
            elif tok == 'REST':
                if nopain_mode != 'baseline_rest':
                    continue
                wins_per_signal = {s: _windows(_col_signal(dfs[s], c)) for s in signals}
                lab = 0
            elif tok is None:
                wins_per_signal = {s: [_col_signal(dfs[s], c)] for s in signals}
                lab = None
            else:
                continue
            k = min((len(wins_per_signal[s]) for s in signals))
            for w in range(k):
                segments.append({s: wins_per_signal[s][w] for s in signals})
                subjects.append(subj)
                labels.append(lab)
                seg_ids.append(c if k == 1 else f'{c}#w{w}')
    labels = None if any((l is None for l in labels)) else labels
    return (segments, subjects, labels, seg_ids)
normalize_per_subject = ai4pain.normalize_per_subject
build_features = ai4pain.build_features
feature_names = ai4pain.feature_names



class AI4Pain2025Dataset:
    """AI4Pain 2025: segments, per-subject normalization and features."""
    load_segments = staticmethod(load_segments)
    normalize_per_subject = staticmethod(normalize_per_subject)
    build_features = staticmethod(build_features)
