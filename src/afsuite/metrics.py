"""PhysioNet/CinC 2017 scoring. Per-class F1 = 2 x correct / (reference count + predicted count);
the official score is the mean F1 of Normal, AF and Other (Noisy is reported but not averaged).
Checked against the published leaderboard: (0.903 + 0.8547 + 0.7366) / 3 = 0.83 for the winning entry."""
import numpy as np
import pandas as pd

from .data import CLASSES


def challenge_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1 = {}
    for c in CLASSES:
        denom = np.sum(y_true == c) + np.sum(y_pred == c)
        f1[c] = 2 * np.sum((y_true == c) & (y_pred == c)) / denom if denom else np.nan
    return float(np.mean([f1["N"], f1["A"], f1["O"]])), f1


def confusion(y_true, y_pred):
    return (pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="predicted"))
            .reindex(index=CLASSES, columns=CLASSES, fill_value=0))
