import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from afsuite.external import DS1, DS2, af_fraction, window_metrics
from afsuite.splits import make_split

SPLIT = os.path.join(os.path.dirname(__file__), "..", "splits", "cinc2017_split.csv")


def test_locked_split_is_stratified_and_disjoint():
    s = pd.read_csv(SPLIT, dtype={"record": str, "label": str})
    assert len(s) == 8528 and s["record"].is_unique
    assert dict(s["label"].value_counts()) == {"N": 5076, "O": 2415, "A": 758, "~": 279}
    test = s["split"] == "test"
    assert test.sum() == 1706 and (s.loc[test, "fold"] == -1).all()
    assert sorted(s.loc[~test, "fold"].unique()) == [0, 1, 2, 3, 4]
    share = s[test]["label"].value_counts() / s["label"].value_counts()
    assert ((share - 0.2).abs() < 0.01).all()


def test_make_split_keeps_duplicates_together():
    labels = np.array(["N"] * 60 + ["A"] * 20 + ["O"] * 15 + ["~"] * 5)
    hashes = [f"h{i}" for i in range(100)]
    hashes[3] = hashes[7] = "same"
    s = make_split([f"R{i}" for i in range(100)], labels, hashes)
    assert s.loc[3, "split"] == s.loc[7, "split"]
    assert (s["fold"] == -1).sum() == (s["split"] == "test").sum()


def test_external_helpers():
    assert set(DS1).isdisjoint(DS2) and len(DS1) == len(DS2) == 22
    iv = [(0.0, 45.0, "(N"), (45.0, 100.0, "(AFIB"), (100.0, 120.0, "(AFL")]
    assert np.allclose(af_fraction([0, 15, 30, 60, 90], iv), [0, 0, 0.5, 1.0, 1 / 3])
    m = window_metrics(np.array([0.9, 0.6, 0.1, 0.0]), np.array(["A", "N", "A", "~"]))
    assert m["windows"] == 4 and m["AF windows"] == 2
    assert m["Se %"] == 50 and m["PPV %"] == 50 and m["Specificity %"] == 50 and m["called Noisy %"] == 25
