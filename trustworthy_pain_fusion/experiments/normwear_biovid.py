"""Frozen NormWear probe on BioVid Part A."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.data import biovid
_PCA_DIM = 120

def _probe():
    return make_pipeline(StandardScaler(), PCA(n_components=_PCA_DIM, random_state=0), LogisticRegression(max_iter=5000, class_weight='balanced'))
from trustworthy_pain_fusion.methods.encoders import load_encoder
from trustworthy_pain_fusion.methods.splits import grouped_kfold, loso_splits
ROOT = _PROJECT_ROOT
NW_CHANNELS = ['gsr', 'ecg', 'emg_trapezius']
EMB_CACHE = ROOT / 'cache' / 'biovid_normwear_emb.npz'

def _kfold_acc(X, y, subj, k=5, seed=0):
    y = np.asarray(y).astype(int)
    accs = []
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        clf = _probe().fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return (float(np.mean(accs)), float(np.std(accs)), len(accs))

def _loso_acc(X, y, subj):
    y = np.asarray(y).astype(int)
    subj = np.asarray(subj)
    accs = []
    for tr, te in loso_splits(subj):
        clf = _probe().fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))
    return (float(np.mean(accs)), float(np.std(accs)), len(accs))

def _eval(X, y, subj, label):
    m = (y == 0) | (y == 4)
    kf_p = _kfold_acc(X[m], (y[m] == 4).astype(int), subj[m])
    lo_p = _loso_acc(X[m], (y[m] == 4).astype(int), subj[m])
    kf_i = _kfold_acc(X, y, subj)
    lo_i = _loso_acc(X, y, subj)
    print(f'[{label}] n={len(set(subj))} | presence 5fold {kf_p[0]:.3f}±{kf_p[1]:.3f} LOSO {lo_p[0]:.3f}±{lo_p[1]:.3f} | intensity 5fold {kf_i[0]:.3f} LOSO {lo_i[0]:.3f}', flush=True)
    mk = lambda t, ch: {'acc': round(t[0], 3), 'std': round(t[1], 3), 'n_folds': t[2], 'chance': ch}
    return {'n_subjects': int(len(set(subj))), 'presence_BL1_vs_PA4': {'kfold5': mk(kf_p, 0.5), 'loso': mk(lo_p, 0.5)}, 'intensity_5class': {'kfold5': mk(kf_i, 0.2), 'loso': mk(lo_i, 0.2)}}

def _get_embeddings():
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=True)
        print(f"loaded cached embeddings {z['X'].shape} from {EMB_CACHE.name}", flush=True)
        return (z['X'], z['y'], z['subj'].astype(str))
    print('loading BioVid Part A (8700 trials)...', flush=True)
    segs, subj, lab, _ = biovid.load_segments(channels=NW_CHANNELS)
    segs = biovid.normalize_per_subject(segs, subj, channels=NW_CHANNELS)
    y = np.asarray(lab, int)
    subj = np.asarray(subj)
    enc = load_encoder('normwear', batch_size=16)
    print(f'encoder={enc.name} dim/ch={enc.dim} dev={enc.device}; embedding {len(segs)} trials x {len(NW_CHANNELS)} channels (src_fs={biovid.FS})...', flush=True)
    X = enc.embed_multivariate(segs, NW_CHANNELS, src_fs=biovid.FS)
    print('embeddings:', X.shape, 'finite', bool(np.isfinite(X).all()), flush=True)
    np.savez(EMB_CACHE, X=X, y=y, subj=subj)
    return (X, y, subj)

def main():
    X, y, subj = _get_embeddings()
    out = {'dataset': 'BioVid Part A (biosignals_filtered)', 'encoder': 'normwear (frozen)', 'channels': NW_CHANNELS, 'embedding_dim': int(X.shape[1]), 'protocol': 'subject-grouped 5-fold AND LOSO logreg on frozen NormWear embeds; per-subject z-norm before encoding', 'subsets': {}}
    out['subsets']['all_87'] = _eval(X, y, subj, 'all-87')
    keep = ~np.isin(subj, list(biovid.weak_responders()))
    out['subsets']['reduced_67'] = _eval(X[keep], y[keep], subj[keep], 'reduced-67')
    out['subsets']['reduced_67']['cite'] = 'werner2017expressiveness (ACIIW 2017)'
    dst = ROOT / 'results' / 'biovid_normwear.json'
    dst.write_text(json.dumps(out, indent=2))
    print('wrote', dst, flush=True)
if __name__ == '__main__':
    main()
