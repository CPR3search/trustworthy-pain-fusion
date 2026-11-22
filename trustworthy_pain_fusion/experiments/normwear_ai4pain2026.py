"""Frozen NormWear probe on AI4Pain 2026."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
from trustworthy_pain_fusion.methods.splits import grouped_kfold
from trustworthy_pain_fusion.methods.encoders import load_encoder
ROOT = _PROJECT_ROOT

def _cv_accuracy(X, y, subj, k=5, seed=0):
    y = np.asarray(y).astype(int)
    accs, Y, P = ([], [], [])
    for tr, te in grouped_kfold(subj, k=k, seed=seed):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        accs.append(float((pred == y[te]).mean()))
        Y.append(y[te])
        P.append(pred)
    bacc = balanced_accuracy_score(np.concatenate(Y), np.concatenate(P))
    return (float(np.mean(accs)), float(np.std(accs)), float(bacc))

def main():
    signals = ai4pain.SIGNALS
    segs, subj, lab, _ = ai4pain.load_segments('data', 'train', signals)
    segs = ai4pain.normalize_per_subject(segs, subj, signals)
    y = np.asarray(lab, int)
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    enc = load_encoder('normwear', batch_size=16)
    print(f'encoder={enc.name} dim/ch={enc.dim} device={enc.device}; embedding {len(segs)} segments x {len(signals)} channels ...')
    X = enc.embed_multivariate(segs, signals, src_fs=ai4pain.FS)
    print('embeddings:', X.shape, 'finite', bool(np.isfinite(X).all()))
    pain = y != 0
    y_pres = (y != 0).astype(int)
    pres_acc, pres_sd, pres_bacc = _cv_accuracy(X, y_pres, si)
    loc_acc, loc_sd, loc_bacc = _cv_accuracy(X[pain], (y[pain] == 2).astype(int), si[pain])
    majority = float(max(y_pres.mean(), 1 - y_pres.mean()))
    out = {'encoder': enc.name, 'checkpoint': str(Path(enc.weights_path).name), 'n_segments': int(len(segs)), 'n_channels': len(signals), 'embedding_dim': int(X.shape[1]), 'per_channel_dim': int(enc.dim), 'presence_acc': round(pres_acc, 3), 'presence_std': round(pres_sd, 3), 'presence_majority_rate': round(majority, 3), 'presence_bacc': round(pres_bacc, 3), 'presence_chance': 0.5, 'location_acc': round(loc_acc, 3), 'location_std': round(loc_sd, 3), 'location_bacc': round(loc_bacc, 3), 'location_chance': 0.5, 'chance_note': 'chance is 0.5 for balanced accuracy; read plain presence accuracy against presence_majority_rate', 'protocol': 'subject-grouped 5-fold logistic regression on frozen NormWear embeddings; per-subject z-norm before encoding; balanced accuracy on pooled out-of-fold predictions'}
    dst = ROOT / 'results' / 'normwear_ai4pain_probe.json'
    dst.write_text(json.dumps(out, indent=2))
    print(f'presence acc {pres_acc:.3f}+/-{pres_sd:.3f} (majority {majority:.3f}), bal.acc {pres_bacc:.3f} | location acc {loc_acc:.3f}+/-{loc_sd:.3f}, bal.acc {loc_bacc:.3f} (chance .50)')
    print('wrote', dst)
if __name__ == '__main__':
    main()
