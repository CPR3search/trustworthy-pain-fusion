"""Classification metrics."""
import numpy as np

def compute_metrics(y_true, y_pred, n_classes=3):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out = {'accuracy': float((y_true == y_pred).mean())}
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[f'precision_{c}'] = p
        out[f'recall_{c}'] = r
        out[f'f1_{c}'] = f
        f1s.append(f)
    out['macro_f1'] = float(np.mean(f1s))
    t_bin, p_bin = ((y_true != 0).astype(int), (y_pred != 0).astype(int))
    out['pain_detection_acc'] = float((t_bin == p_bin).mean())
    true_pain = y_true != 0
    out['loc_acc_among_truepain'] = float((y_pred[true_pain] == y_true[true_pain]).mean()) if true_pain.any() else 0.0
    detected = true_pain & (y_pred != 0)
    out['arm_vs_hand_acc'] = float((y_pred[detected] == y_true[detected]).mean()) if detected.any() else 0.0
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return (out, cm)
