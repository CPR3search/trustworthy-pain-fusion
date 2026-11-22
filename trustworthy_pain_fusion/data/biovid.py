"""BioVid Heat Pain Part A loader, per-subject normalization and features."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
from pathlib import Path
import numpy as np
import pandas as pd
from trustworthy_pain_fusion.data.ai4pain2026 import _generic, _GENERIC_NAMES, _hjorth, _HJORTH_NAMES, _spectral, _SPECTRAL_NAMES, _dynamics, _DYNAMICS_NAMES, _eda, _EDA_NAMES, _bvp, _BVP_NAMES
FS = 512
DATA_DIR = 'data/biovid/PartA/biosignals_filtered'
CHANNELS = ['gsr', 'ecg', 'emg_trapezius', 'emg_corrugator', 'emg_zygomaticus']
CLASS_MAP = {'BL1': 0, 'PA1': 1, 'PA2': 2, 'PA3': 3, 'PA4': 4}
CLASS_NAMES = ['BL1', 'PA1', 'PA2', 'PA3', 'PA4']
_SPECIALIZED = {'gsr': (_eda, _EDA_NAMES), 'ecg': (_bvp, _BVP_NAMES)}
_SHARED_NAMES = _GENERIC_NAMES + _HJORTH_NAMES + _SPECTRAL_NAMES + _DYNAMICS_NAMES

def parse_class(filename: str):
    parts = Path(filename).stem.split('-')
    return CLASS_MAP.get(parts[1]) if len(parts) >= 2 else None

def load_segments(data_dir=DATA_DIR, channels=CHANNELS, subjects=None, per_class=None):
    base = Path(data_dir)
    subj_dirs = sorted((d for d in base.iterdir() if d.is_dir()))
    if subjects is not None:
        keep = set(subjects)
        subj_dirs = [d for d in subj_dirs if d.name in keep]
    if not subj_dirs:
        raise FileNotFoundError(f'no subject dirs under {base.resolve()}')
    segments, subj_out, labels, seg_ids = ([], [], [], [])
    for d in subj_dirs:
        counts: dict[int, int] = {}
        for f in sorted(d.glob('*_bio.csv')):
            lab = parse_class(f.name)
            if lab is None:
                continue
            if per_class is not None and counts.get(lab, 0) >= per_class:
                continue
            counts[lab] = counts.get(lab, 0) + 1
            df = pd.read_csv(f, sep='\t')
            seg = {c: df[c].to_numpy(dtype=float) for c in channels}
            segments.append(seg)
            subj_out.append(d.name)
            labels.append(lab)
            seg_ids.append(f.stem.replace('_bio', ''))
    return (segments, subj_out, labels, seg_ids)

def normalize_per_subject(segments, subjects, channels=CHANNELS):
    subjects = np.asarray(subjects)
    out = [dict(seg) for seg in segments]
    for subj in np.unique(subjects):
        idx = np.where(subjects == subj)[0]
        for c in channels:
            allvals = np.concatenate([segments[i][c] for i in idx])
            mu, sd = (allvals.mean(), allvals.std() + 1e-08)
            for i in idx:
                out[i][c] = (segments[i][c] - mu) / sd
    return out

def feature_names(channels=CHANNELS):
    names = []
    for c in channels:
        names += [f'{c}_{n}' for n in _SHARED_NAMES]
        if c in _SPECIALIZED:
            names += [f'{c}_{n}' for n in _SPECIALIZED[c][1]]
    return names

def segment_features(seg, channels=CHANNELS, fs=FS):
    feats = []
    for c in channels:
        x = seg[c]
        feats += _generic(x)
        feats += _hjorth(x)
        feats += _spectral(x, fs=fs)
        feats += _dynamics(x)
        if c in _SPECIALIZED:
            fn, _ = _SPECIALIZED[c]
            feats += fn(x, fs=fs)
    return np.asarray(feats, float)

def build_features(segments, channels=CHANNELS, fs=FS):
    X = np.vstack([segment_features(seg, channels, fs) for seg in segments])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

def biovid_targets(y):
    y = np.asarray(y)
    bl_pa4 = (y == 0) | (y == 4)
    return {'presence (BL1 vs PA4)': ((y[bl_pa4] == 4).astype(int), bl_pa4), 'intensity (BL1..PA4)': (y.astype(int), np.ones(len(y), bool))}

def weak_responders():
    """Subject IDs excluded from the 67-subject BioVid subset (Werner et al., 2017).

    The list is read from data/biovid/excluded_subjects.txt (one ID per line), which the user
    prepares from the published subset definition; it is not distributed with this code.
    """
    path = _PROJECT_ROOT / 'data' / 'biovid' / 'excluded_subjects.txt'
    if not path.exists():
        raise FileNotFoundError(f'{path} is required for the 67-subject BioVid subset')
    ids = tuple(path.read_text().split())
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError(f'{path} must list 20 distinct subject IDs, found {len(set(ids))}')
    known = {p.name for p in Path(DATA_DIR).iterdir() if p.is_dir() and any(p.glob('*_bio.csv'))}
    unknown = sorted(set(ids) - known)
    if unknown:
        raise ValueError(f'{path} lists IDs that are not BioVid Part A subjects: {unknown}')
    return ids



class BioVidDataset:
    """BioVid Heat Pain Part A: segments, per-subject normalization and features."""
    channels = CHANNELS
    load_segments = staticmethod(load_segments)
    normalize_per_subject = staticmethod(normalize_per_subject)
    build_features = staticmethod(build_features)
