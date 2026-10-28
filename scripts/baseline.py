"""Baseline: gradient boosting on the 35 recording features. The feature set is chosen by 5-fold cross-validation
on the training part; the chosen model is then trained on all training recordings and scored once on the test set."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from afsuite.config import load_config
from afsuite.data import CLASSES, NAMES
from afsuite.features import RR_KEYS
from afsuite.metrics import challenge_score, confusion


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("features", "split", "baseline_model", "results"):
        p.add_argument(f"--{k}", default=cfg[k])
    a = p.parse_args()
    F = pd.read_csv(a.features)
    split = pd.read_csv(a.split, dtype={"record": str, "label": str})
    assert len(F) == len(split), "features and split have different numbers of recordings"
    y = split["label"].to_numpy()
    train, test = (split["split"] == "train").to_numpy(), (split["split"] == "test").to_numpy()
    fold = split["fold"].to_numpy()

    def model(kw):
        return HistGradientBoostingClassifier(**cfg["baseline"], **kw)

    configs = {"RR features only": (["beats_per_s"] + RR_KEYS, {}),
               "All features": (list(F.columns), {}),
               "All features, class-balanced": (list(F.columns), {"class_weight": "balanced"})}
    rows = []
    for name, (cols, kw) in configs.items():
        s, per = [], []
        for k in range(cfg["n_folds"]):
            tr, va = train & (fold != k), fold == k
            pred = model(kw).fit(F.loc[tr, cols], y[tr]).predict(F.loc[va, cols])
            sc, f1 = challenge_score(y[va], pred)
            s.append(sc)
            per.append(f1)
        per = pd.DataFrame(per).mean()
        row = {"model": name, "CV score": f"{np.mean(s):.3f} ± {np.std(s, ddof=1):.3f}", "_mean": np.mean(s)}
        row.update({f"F1 {NAMES[c]}": round(per[c], 3) for c in CLASSES})
        rows.append(row)
    cv = pd.DataFrame(rows)
    print("5-fold CV on the training part (official score = mean F1 of Normal, AF and Other):")
    print(cv.drop(columns="_mean").to_string(index=False))
    best = cv.loc[cv["_mean"].idxmax(), "model"]
    os.makedirs(a.results, exist_ok=True)
    cv.drop(columns="_mean").to_csv(os.path.join(a.results, "baseline_cv.csv"), index=False)

    cols, kw = configs[best]
    clf = model(kw).fit(F.loc[train, cols], y[train])
    pred = clf.predict(F.loc[test, cols])
    score, f1 = challenge_score(y[test], pred)
    print(f"\nChosen by CV: '{best}'. Test set ({test.sum():,} recordings): official score {score:.3f} | "
          + " · ".join(f"F1 {NAMES[c]} {f1[c]:.3f}" for c in CLASSES))
    cm = confusion(y[test], pred)
    print(cm.to_string())
    os.makedirs(os.path.dirname(os.path.abspath(a.baseline_model)), exist_ok=True)
    joblib.dump({"model": clf, "features": cols}, a.baseline_model)
    row = {"model": best, "test score": round(score, 4)}
    row.update({f"F1 {NAMES[c]}": round(f1[c], 4) for c in CLASSES})
    pd.DataFrame([row]).to_csv(os.path.join(a.results, "baseline_test.csv"), index=False)
    cm.to_csv(os.path.join(a.results, "baseline_confusion.csv"))


if __name__ == "__main__":
    main()
