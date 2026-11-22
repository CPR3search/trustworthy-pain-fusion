"""Classifier factory used by the benchmark pipeline."""
import numpy as np
from trustworthy_pain_fusion.methods.registry import build_model, register

@register('nearest_centroid')
class NearestCentroid:

    def __init__(self, normalize: bool=True):
        self.normalize = normalize
        self.mu = self.sd = None
        self.centroids = {}

    def _norm(self, X, fit):
        if not self.normalize:
            return X
        if fit:
            self.mu, self.sd = (X.mean(0), X.std(0) + 1e-08)
        return (X - self.mu) / self.sd

    def fit(self, X, y, subjects=None):
        Xn = self._norm(X, fit=True)
        self.centroids = {c: Xn[y == c].mean(0) for c in np.unique(y)}
        return self

    def predict(self, X):
        Xn = self._norm(X, fit=False)
        classes = sorted(self.centroids)
        d = np.stack([((Xn - self.centroids[c]) ** 2).sum(1) for c in classes], 1)
        return np.array(classes)[d.argmin(1)]

@register('random_forest')
class RandomForestModel:

    def __init__(self, n_estimators=400, max_depth=None, min_samples_leaf=1, class_weight=None, scale=False, seed=0):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight=class_weight, random_state=seed, n_jobs=-1)
        self.scale = scale
        self.scaler = None

    def fit(self, X, y, subjects=None):
        if self.scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler().fit(X)
            X = self.scaler.transform(X)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.clf.predict(X)

@register('svm')
class SVMModel:

    def __init__(self, C=10.0, gamma='scale', kernel='rbf', class_weight='balanced', seed=0):
        from sklearn.svm import SVC
        self.clf = SVC(C=C, gamma=gamma, kernel=kernel, class_weight=class_weight, random_state=seed)
        self.scaler = None

    def fit(self, X, y, subjects=None):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(X)
        self.clf.fit(self.scaler.transform(X), y)
        return self

    def predict(self, X):
        return self.clf.predict(self.scaler.transform(X))

@register('lgbm')
class LGBMModel:

    def __init__(self, n_estimators=400, num_leaves=31, learning_rate=0.05, max_depth=-1, class_weight='balanced', subsample=0.9, colsample_bytree=0.9, n_jobs=4, seed=0):
        from lightgbm import LGBMClassifier
        self.clf = LGBMClassifier(n_estimators=n_estimators, num_leaves=num_leaves, learning_rate=learning_rate, max_depth=max_depth, class_weight=class_weight, subsample=subsample, colsample_bytree=colsample_bytree, random_state=seed, n_jobs=n_jobs, verbose=-1)

    def fit(self, X, y, subjects=None):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)
