"""Subject-grouped and leave-one-subject-out splits."""
from typing import Iterator
import numpy as np

def loso_splits(subjects: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    for s in np.unique(subjects):
        test = np.where(subjects == s)[0]
        train = np.where(subjects != s)[0]
        yield (train, test)

def grouped_kfold(subjects: np.ndarray, k: int, seed: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, k)
    for f in folds:
        test = np.isin(subjects, f)
        yield (np.where(~test)[0], np.where(test)[0])
