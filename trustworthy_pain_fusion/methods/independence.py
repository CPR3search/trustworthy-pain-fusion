"""Predictor-independent tests: C2ST, MMD and k-NN error."""
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import argparse
import json
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from trustworthy_pain_fusion.methods.decodability import N_JOBS, null_labels
ROOT = _PROJECT_ROOT
RESULTS = ROOT / 'results'
EPS = 1e-12

def mine_mi_bits(X, y, subj, epochs=200, hidden=128, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = StandardScaler().fit_transform(np.asarray(X, float))
    y = np.asarray(y).astype(int)
    subj = np.asarray(subj)
    uniq = rng.permutation(np.unique(subj))
    cut = max(1, int(0.7 * len(uniq)))
    tr_s, ev_s = (set(uniq[:cut]), set(uniq[cut:]))
    tr = np.isin(subj, list(tr_s))
    ev = np.isin(subj, list(ev_s))
    if not ev.any() or len(np.unique(y[tr])) < 2:
        tr = ev = np.ones(len(y), bool)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=dev)
    d = X.shape[1]
    net = nn.Sequential(nn.Linear(d + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    tr_i = np.where(tr)[0]
    ev_i = np.where(ev)[0]

    def dv_bound(idx):
        xb = Xt[idx]
        yb = yt[idx]
        yb_shuf = yb[torch.randperm(len(idx), device=dev)]
        t_joint = net(torch.cat([xb, yb], 1))
        t_marg = net(torch.cat([xb, yb_shuf], 1))
        return t_joint.mean() - torch.log(torch.exp(t_marg).mean() + EPS)
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = -dv_bound(tr_i)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        mi_nats = float(dv_bound(ev_i).item())
    return max(0.0, mi_nats) / np.log(2)

def _cv_auc(X, yy, subj, k, seed, rf_jobs):
    aucs = []
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(X, yy, subj):
        if len(np.unique(yy[tr])) < 2 or len(np.unique(yy[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = RandomForestClassifier(n_estimators=200, n_jobs=rf_jobs, random_state=seed)
        clf.fit(sc.transform(X[tr]), yy[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(yy[te], p))
    return float(np.mean(aucs)) if aucs else 0.5

def c2st(X, y, subj, k=5, seed=0, n_perm=100, perm_scheme='within', n_jobs=None):
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    subj = np.asarray(subj)
    n_jobs = N_JOBS if n_jobs is None else n_jobs
    obs = _cv_auc(X, y, subj, k, seed, 4)
    perms = null_labels(y, subj, n_perm, seed=seed, scheme=perm_scheme)
    if n_jobs > 1:
        null = np.array(Parallel(n_jobs=n_jobs)((delayed(_cv_auc)(X, yp, subj, k, seed, 1) for yp in perms)))
    else:
        null = np.array([_cv_auc(X, yp, subj, k, seed, 4) for yp in perms])
    p_val = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {'auc': round(obs, 4), 'auc_perm_mean': round(float(null.mean()), 4), 'p_value': round(p_val, 4), 'n_perm': n_perm, 'perm_scheme': perm_scheme}

def mmd_test(X, y, subj, seed=0, n_perm=300, max_n=1200, perm_scheme='within'):
    X = StandardScaler().fit_transform(np.asarray(X, float))
    y = np.asarray(y).astype(int)
    subj = np.asarray(subj)
    rng = np.random.default_rng(seed)
    ia = np.flatnonzero(y == 0)
    ib = np.flatnonzero(y == 1)
    if len(ia) > max_n:
        ia = rng.choice(ia, max_n, replace=False)
    if len(ib) > max_n:
        ib = rng.choice(ib, max_n, replace=False)
    idx = np.concatenate([ia, ib])
    Z, lab, ss = (X[idx], y[idx], subj[idx])
    s = Z[rng.choice(len(Z), min(500, len(Z)), replace=False)]
    med = np.median(euclidean_distances(s, s)) + EPS
    gamma = 1.0 / (2 * med ** 2)
    K = rbf_kernel(Z, Z, gamma)
    np.fill_diagonal(K, 0.0)

    def mmd2(l):
        a = (l == 0).astype(float)
        b = 1.0 - a
        na, nb = (a.sum(), b.sum())
        Ka, Kb = (K @ a, K @ b)
        return float(a @ Ka / (na * (na - 1) + EPS) + b @ Kb / (nb * (nb - 1) + EPS) - 2.0 * (b @ Ka) / (na * nb + EPS))
    obs = mmd2(lab)
    null = np.array([mmd2(lp) for lp in null_labels(lab, ss, n_perm, seed=seed + 1, scheme=perm_scheme)])
    p_val = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {'mmd2': round(obs, 6), 'mmd2_perm_mean': round(float(null.mean()), 6), 'p_value': round(p_val, 4), 'n_perm': n_perm, 'perm_scheme': perm_scheme}

def knn_bayes_floor(X, y, subj, k=5, seed=0, n_proj=3, proj_dim=16):
    X = np.asarray(X, float)
    y = np.asarray(y).astype(int)
    subj = np.asarray(subj)
    rng = np.random.default_rng(seed)

    def grouped_knn_err(Xx):
        errs = []
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for tr, te in sgkf.split(Xx, y, subj):
            sc = StandardScaler().fit(Xx[tr])
            clf = KNeighborsClassifier(n_neighbors=15)
            clf.fit(sc.transform(Xx[tr]), y[tr])
            errs.append(1.0 - float((clf.predict(sc.transform(Xx[te])) == y[te]).mean()))
        return float(np.mean(errs))
    feat_err = grouped_knn_err(X)
    proj_errs = []
    for _ in range(n_proj):
        P = rng.normal(size=(X.shape[1], proj_dim))
        proj_errs.append(grouped_knn_err(X @ P))
    maj = 1.0 - float(np.bincount(y).max() / len(y))
    return {'knn_error_features': round(feat_err, 4), 'knn_error_randproj_mean': round(float(np.mean(proj_errs)), 4), 'chance_error': round(maj, 4)}

def assess_binary(X, y, subj, label, do_mine=False, c2st_perm=100, mmd_perm=1000):
    y = np.asarray(y).astype(int)
    res = {'label': label, 'n': int(len(y)), 'pos_rate': round(float(y.mean()), 3)}
    res['c2st'] = c2st(X, y, subj, n_perm=c2st_perm)
    res['mmd'] = mmd_test(X, y, subj, n_perm=mmd_perm)
    res['knn_bayes'] = knn_bayes_floor(X, y, subj)
    if do_mine:
        try:
            res['mine_mi_bits'] = round(mine_mi_bits(X, y, subj), 4)
        except Exception as e:
            res['mine_mi_bits'] = None
            res['mine_error'] = f'{type(e).__name__}: {e}'
    one = lambda m, k: res[m][k]
    print(f"[{label}] C2ST AUC {one('c2st', 'auc')} (p={one('c2st', 'p_value')}) | MMD2 {one('mmd', 'mmd2')} (p={one('mmd', 'p_value')}) | kNN err {one('knn_bayes', 'knn_error_features')} (chance {one('knn_bayes', 'chance_error')})", flush=True)
    return res

def _binary_contrasts():
    out = []
    from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
    segs, subj, lab, _ = ai4pain.load_segments('data', 'train', ai4pain.SIGNALS)
    segs = ai4pain.normalize_per_subject(segs, subj, ai4pain.SIGNALS)
    X = ai4pain.build_features(segs, ai4pain.SIGNALS)
    y = np.asarray(lab, int)
    s = np.asarray(subj)
    out.append(('ai4pain2026', 'presence_NoPain_vs_Pain', X, (y != 0).astype(int), s))
    pain = y != 0
    out.append(('ai4pain2026', 'LOCATION_Arm_vs_Hand', X[pain], (y[pain] == 2).astype(int), s[pain]))
    from trustworthy_pain_fusion.data import ai4pain2025 as a25
    segs2, subj2, lab2, _ = a25.load_segments(split='train', nopain_mode='baseline_seg')
    segs2 = a25.normalize_per_subject(segs2, subj2)
    X2 = a25.build_features(segs2)
    y2 = np.asarray(lab2, int)
    s2 = np.asarray(subj2)
    out.append(('ai4pain2025', 'presence_NoPain_vs_Pain', X2, (y2 != 0).astype(int), s2))
    pain2 = y2 != 0
    out.append(('ai4pain2025', 'INTENSITY_Low_vs_High', X2[pain2], (y2[pain2] == 2).astype(int), s2[pain2]))
    from trustworthy_pain_fusion.data import biovid
    segsb, subjb, labb, _ = biovid.load_segments()
    segsb = biovid.normalize_per_subject(segsb, subjb)
    Xb = biovid.build_features(segsb)
    yb = np.asarray(labb, int)
    sb = np.asarray(subjb)
    m = (yb == 0) | (yb == 4)
    out.append(('biovid', 'presence_BL1_vs_PA4', Xb[m], (yb[m] == 4).astype(int), sb[m]))
    return out

def main():
    results = {}
    for dataset, name, X, y, subj in _binary_contrasts():
        perm = 1000 if 'LOCATION' in name else 100
        results.setdefault(dataset, {})[name] = assess_binary(X, y, subj, f'{dataset}:{name}', c2st_perm=perm)
    results['protocol'] = 'subject-grouped CV; C2ST and MMD permutation nulls shuffle labels within each subject'
    dst = RESULTS / 'model_independence.json'
    dst.write_text(json.dumps(results, indent=2))
    print('wrote', dst, flush=True)
if __name__ == '__main__':
    main()


class PredictorIndependentTests:
    """Classifier two-sample test, kernel MMD test and k-NN error for a binary contrast."""
    c2st = staticmethod(c2st)
    mmd = staticmethod(mmd_test)
    knn_error = staticmethod(knn_bayes_floor)
    assess = staticmethod(assess_binary)
