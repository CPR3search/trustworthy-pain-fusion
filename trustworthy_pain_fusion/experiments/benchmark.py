"""Feature pipeline benchmark on AI4Pain 2026."""
import json
from pathlib import Path
import numpy as np
from trustworthy_pain_fusion.data import ai4pain2026 as ai4pain
from trustworthy_pain_fusion.methods.metrics import compute_metrics
from trustworthy_pain_fusion.methods.models import LGBMModel, RandomForestModel, SVMModel
from trustworthy_pain_fusion.methods.registry import build_model
from trustworthy_pain_fusion.methods.splits import loso_splits
MODELS = {'NearestCentroid': {'name': 'nearest_centroid', 'args': {'normalize': True}}, 'RandomForest': {'name': 'random_forest', 'args': {'n_estimators': 500, 'min_samples_leaf': 2, 'class_weight': 'balanced'}}, 'SVM-RBF': {'name': 'svm', 'args': {'C': 10.0, 'gamma': 'scale', 'class_weight': 'balanced'}}, 'LightGBM': {'name': 'lgbm', 'args': {'n_estimators': 600, 'learning_rate': 0.03, 'num_leaves': 31, 'class_weight': 'balanced'}}}

def _feats(split, signals=ai4pain.SIGNALS):
    segs, subj, lab, _ = ai4pain.load_segments('data', split, signals)
    segs = ai4pain.normalize_per_subject(segs, subj, signals)
    X = ai4pain.build_features(segs, signals)
    _, si = np.unique(np.asarray(subj), return_inverse=True)
    return (X, None if lab is None else np.asarray(lab, int), si)

def _loso(X, y, subj, cfg):
    rows = []
    for tr, te in loso_splits(subj):
        m = build_model(cfg).fit(X[tr], y[tr], subj[tr])
        met, _ = compute_metrics(y[te], m.predict(X[te]), 3)
        rows.append(met)
    agg = lambda k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows]))
    return {k: agg(k) for k in ['accuracy', 'pain_detection_acc', 'arm_vs_hand_acc']}

def main():
    Xtr, ytr, subtr = _feats('train')
    Xv, yv, _ = _feats('validation')
    print(f'features: {Xtr.shape[1]}  | train {len(ytr)}  val {len(yv)}\n')
    print(f"{'model':14s} | {'LOSO acc':>14s} {'LOSO det':>9s} {'LOSO a/h':>9s} | {'val acc':>7s} {'val det':>7s} {'val a/h':>7s}")
    print('-' * 86)
    out = {}
    for name, cfg in MODELS.items():
        lo = _loso(Xtr, ytr, subtr, cfg)
        model = build_model(cfg).fit(Xtr, ytr, subtr)
        vmet, _ = compute_metrics(yv, model.predict(Xv), 3)
        out[name] = {'loso': {k: list(v) for k, v in lo.items()}, 'val': vmet}
        print(f"{name:14s} | {lo['accuracy'][0] * 100:6.1f}±{lo['accuracy'][1] * 100:4.1f} {lo['pain_detection_acc'][0] * 100:8.1f} {lo['arm_vs_hand_acc'][0] * 100:8.1f} | {vmet['accuracy'] * 100:6.1f} {vmet['pain_detection_acc'] * 100:6.1f} {vmet['arm_vs_hand_acc'] * 100:6.1f}")
    Path('results/bench.json').write_text(json.dumps(out, indent=2))
    print('\nwrote results/bench.json')
if __name__ == '__main__':
    main()
