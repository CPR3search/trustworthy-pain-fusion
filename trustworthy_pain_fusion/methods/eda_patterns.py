"""Binary-pattern texture features for EDA."""
import numpy as np
from trustworthy_pain_fusion.methods.lbp import _UNI_LUT, _UNI_NBINS
EPS = 1e-12

def _domains(x):
    spec = np.abs(np.fft.fft(x))
    ceps = np.real(np.fft.ifft(np.log(spec + EPS)))
    return {'t': x, 's': spec, 'c': ceps}

def _uni_hist(dec):
    if len(dec) == 0:
        return np.zeros(_UNI_NBINS)
    h = np.bincount(_UNI_LUT[dec], minlength=_UNI_NBINS).astype(float)
    return h / (h.sum() + EPS)

def _hist(dec, nbins):
    if len(dec) == 0:
        return np.zeros(nbins)
    h = np.bincount(dec, minlength=nbins).astype(float)
    return h / (h.sum() + EPS)

def _windows(sig, n):
    if len(sig) < n:
        return None
    return np.lib.stride_tricks.sliding_window_view(sig, n)

def lbp(sig, n=9):
    w = _windows(sig, n)
    if w is None:
        return np.zeros(_UNI_NBINS)
    c = n // 2
    bits = (np.delete(w, c, axis=1) >= w[:, c:c + 1]).astype(np.int64)
    return _uni_hist(bits @ (1 << np.arange(bits.shape[1])))

def lbp_r2(sig, n=17):
    w = _windows(sig, n)
    if w is None:
        return np.zeros(_UNI_NBINS)
    c = n // 2
    idx = [c - 8, c - 6, c - 4, c - 2, c + 2, c + 4, c + 6, c + 8]
    bits = (w[:, idx] >= w[:, c:c + 1]).astype(np.int64)
    return _uni_hist(bits @ (1 << np.arange(8)))

def cslbp(sig, n=9):
    w = _windows(sig, n)
    if w is None:
        return np.zeros(16)
    bits = (w[:, [0, 1, 2, 3]] >= w[:, [8, 7, 6, 5]]).astype(np.int64)
    return _hist(bits @ (1 << np.arange(4)), 16)

def ltp(sig, n=9, t=0.2):
    w = _windows(sig, n)
    if w is None:
        return np.concatenate([np.zeros(_UNI_NBINS), np.zeros(_UNI_NBINS)])
    c = n // 2
    neigh = np.delete(w, c, axis=1)
    diff = neigh - w[:, c:c + 1]
    thr = t * (w.std(axis=1, keepdims=True) + EPS)
    up = (diff >= thr).astype(np.int64) @ (1 << np.arange(8))
    lo = (diff <= -thr).astype(np.int64) @ (1 << np.arange(8))
    return np.concatenate([_uni_hist(up), _uni_hist(lo)])

def glbp(sig, n=9):
    return lbp(np.diff(sig), n)
VARIANTS = {'LBP': lbp, 'LBPR2': lbp_r2, 'CSLBP': cslbp, 'LTP': ltp, 'GLBP': glbp}

def bp_bank(x):
    doms = _domains(x)
    vals, names = ([], [])
    for vname, fn in VARIANTS.items():
        for dname, dsig in doms.items():
            h = fn(dsig)
            vals.append(h)
            names += [f'bp{vname}_{dname}_{i}' for i in range(len(h))]
    return (np.concatenate(vals), names)
