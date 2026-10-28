"""The locked CinC 2017 split: a stratified 20% test set and 5 stratified folds on the rest.
Recordings with identical samples (exact duplicates) are kept on the same side of the train/test split."""
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def recording_hashes(samples, offsets):
    """MD5 of each recording's raw samples, used to find exact duplicates."""
    return [hashlib.md5(samples[offsets[i]:offsets[i + 1]].tobytes()).hexdigest() for i in range(len(offsets) - 1)]


def make_split(records, labels, hashes, test_size=0.2, seed=2017, n_folds=5, fold_seed=0):
    """DataFrame with columns record, label, split ('train' / 'test') and fold (0..n_folds-1; -1 for test)."""
    labels = np.asarray(labels)
    groups = pd.Series(hashes).factorize()[0]
    first = pd.Series(groups).drop_duplicates().index.to_numpy()
    _, test_groups = train_test_split(groups[first], test_size=test_size, stratify=labels[first], random_state=seed)
    split = pd.DataFrame({"record": np.asarray(records), "label": labels})
    split["split"] = np.where(np.isin(groups, test_groups), "test", "train")
    split["fold"] = -1
    tr = np.where(split["split"] == "train")[0]
    folds = StratifiedKFold(n_folds, shuffle=True, random_state=fold_seed)
    for k, (_, va) in enumerate(folds.split(tr, labels[tr])):
        split.loc[tr[va], "fold"] = k
    return split
