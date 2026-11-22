"""Uniform local-binary-pattern lookup table."""
import json
from pathlib import Path
import numpy as np
from scipy import signal as sp_signal
from scipy import stats as sp_stats
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
from trustworthy_pain_fusion.methods.splits import loso_splits

def _uniform_lut():
    lut = np.zeros(256, np.int64)
    uni = []
    for v in range(256):
        b = [v >> i & 1 for i in range(8)]
        trans = sum((b[i] != b[(i + 1) % 8] for i in range(8)))
        uni.append(trans <= 2)
    label, nxt = ({}, 0)
    for v in range(256):
        if uni[v]:
            label[v] = nxt
            nxt += 1
    for v in range(256):
        lut[v] = label[v] if uni[v] else nxt
    return (lut, nxt + 1)
_UNI_LUT, _UNI_NBINS = _uniform_lut()
