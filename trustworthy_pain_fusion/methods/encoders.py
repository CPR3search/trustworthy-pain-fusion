"""Frozen foundation-model encoders (NormWear)."""
from __future__ import annotations
from trustworthy_pain_fusion.paths import PROJECT_ROOT as _PROJECT_ROOT
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample as _resample
SRC_FS = 100

def resample_window(x, dst_fs, dst_len, src_fs=SRC_FS):
    x = np.asarray(x, float)
    n = max(1, int(round(len(x) * dst_fs / src_fs)))
    r = _resample(x, n)
    if len(r) >= dst_len:
        return r[:dst_len]
    out = np.zeros(dst_len)
    out[:len(r)] = r
    return out

class FMEncoder:
    name = 'base'
    target_fs = 100
    target_len = 1000
    dim = 128

    def __init__(self, weights_path=None, device=None, batch_size=64, fp16=True):
        import torch
        self.torch = torch
        self.weights_path = weights_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.fp16 = fp16 and self.device == 'cuda'
        self.net = None

    def _build(self):
        raise NotImplementedError

    def _forward(self, batch):
        raise NotImplementedError

    def load(self):
        self.net = self._build()
        self.net.to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        if self.fp16:
            self.net.half()
        return self

    def embed(self, bvp_segments):
        if self.net is None:
            self.load()
        torch = self.torch
        W = np.stack([resample_window(x, self.target_fs, self.target_len) for x in bvp_segments])
        embs = []
        with torch.no_grad():
            for i in range(0, len(W), self.batch_size):
                b = torch.tensor(W[i:i + self.batch_size], dtype=torch.float32)
                b = b.unsqueeze(1).to(self.device)
                if self.fp16:
                    b = b.half()
                embs.append(np.asarray(self._forward(b), dtype=np.float32))
        return np.concatenate(embs, 0)

class NormWearEncoder(FMEncoder):
    name = 'normwear'
    target_fs = 65
    target_len = 715
    dim = 768
    VENDOR_PARENT = 'third_party'
    DEFAULT_WEIGHTS = 'checkpoints/normwear/normwear_pretrain_ckpt.pth'

    def __init__(self, weights_path=None, device=None, batch_size=16, fp16=False):
        if weights_path is None:
            root = _PROJECT_ROOT
            weights_path = str(root / self.DEFAULT_WEIGHTS)
        super().__init__(weights_path=weights_path, device=device, batch_size=batch_size, fp16=fp16)

    def _build(self):
        parent = str(_PROJECT_ROOT / self.VENDOR_PARENT)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        from NormWear.main_model import NormWearModel
        return NormWearModel(weight_path=self.weights_path, optimized_cwt=True)

    def embed(self, *args, **kwargs):
        raise TypeError('NormWear is multivariate; call embed_multivariate(segments, signals, src_fs=...) instead of the single-channel embed().')

    def embed_multivariate(self, segments, signals, src_fs=SRC_FS):
        if self.net is None:
            self.load()
        torch = self.torch
        W = np.stack([np.stack([resample_window(seg[s], self.target_fs, self.target_len, src_fs) for s in signals]) for seg in segments])
        embs = []
        with torch.no_grad():
            for i in range(0, len(W), self.batch_size):
                b = torch.tensor(W[i:i + self.batch_size], dtype=torch.float32)
                out = self.net.get_embedding(b, sampling_rate=self.target_fs, device=self.device)
                e = out[:, :, 0, :]
                e = e.reshape(e.shape[0], -1)
                embs.append(np.asarray(e.detach().float().cpu().numpy(), dtype=np.float32))
                del out, e
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
        return np.concatenate(embs, 0)
_REGISTRY = {'normwear': NormWearEncoder}

def load_encoder(name='normwear', **kw):
    if name not in _REGISTRY:
        raise KeyError(f"unknown encoder '{name}'; options {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kw).load()

