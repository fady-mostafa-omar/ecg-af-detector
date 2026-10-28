"""Train the hybrid CNN (30-s windows + window features) with several seeds and save the ensemble.
Training uses folds 1-4 of the training part; fold 0 chooses the epoch. The test set is not touched.
A GPU is strongly recommended (about 2 minutes per seed on a Colab T4; several hours per seed on a CPU).
Each seed is saved as soon as it finishes, so a re-run resumes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import time

import numpy as np
import pandas as pd
import torch

from afsuite.cnn import FeatureScaler, load_members, save_members, train
from afsuite.config import load_config
from afsuite.data import CLASSES


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("windows", "window_features", "split", "model", "seed_dir"):
        p.add_argument(f"--{k}", default=cfg[k])
    p.add_argument("--seeds", type=int, nargs="+", default=cfg["seeds"])
    p.add_argument("--epochs", type=int, default=cfg["epochs"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None,
                   help="use only the first N training and N validation windows (quick check, not for real use)")
    a = p.parse_args()
    print(f"Device: {a.device}" + ("" if a.device == "cuda" else "  (no GPU: full training takes hours)"))

    split = pd.read_csv(a.split, dtype={"record": str, "label": str})
    labels = split["label"].to_numpy()
    train_rec = ((split["split"] == "train") & (split["fold"] != cfg["val_fold"])).to_numpy()
    val_rec = (split["fold"] == cfg["val_fold"]).to_numpy()
    z = np.load(a.windows)
    waves, owner = z["waves"].astype(np.float32), z["owner"]
    WF = pd.read_csv(a.window_features)
    assert owner.max() < len(split), "windows and split do not match"
    tr_idx, va_idx = np.where(train_rec[owner])[0], np.where(val_rec[owner])[0]
    scaler = FeatureScaler().fit(WF.to_numpy(float)[tr_idx])         # statistics of the training windows only
    if a.limit:
        tr_idx, va_idx = tr_idx[:a.limit], va_idx[:a.limit]
    feats = scaler(WF.to_numpy(float))
    y_win = np.array([CLASSES.index(c) for c in labels])[owner]
    val_recs = np.unique(owner[va_idx])
    print(f"training windows {len(tr_idx):,} | validation windows {len(va_idx):,} ({len(val_recs):,} recordings)")

    os.makedirs(a.seed_dir, exist_ok=True)
    members = []
    for s in a.seeds:
        path = os.path.join(a.seed_dir, f"seed{s}.pt")
        if os.path.exists(path):
            m, sc, ck = load_members(path)
            assert np.allclose(sc.mu, scaler.mu, equal_nan=True), f"{path} was trained on other data; delete it"
            members.append(m[0])
            print(f"seed {s}: loaded (best epoch {ck['best_epoch']}, validation {ck['best_val']:.3f})")
            continue
        t0 = time.time()
        model, best, ep = train(waves, feats, y_win, tr_idx, va_idx, owner, labels, val_recs, epochs=a.epochs,
                                batch=cfg["batch_size"], lr=cfg["learning_rate"], seed=s, device=a.device,
                                log=lambda msg, s=s: print(f"   seed {s} | {msg}", flush=True))
        save_members(path, [model], scaler, {"seed": s, "best_epoch": ep, "best_val": best})
        members.append(model)
        print(f"seed {s}: best epoch {ep}, validation mean F1 (4 classes) {best:.3f} | {time.time() - t0:.0f} s")
    os.makedirs(os.path.dirname(os.path.abspath(a.model)), exist_ok=True)
    save_members(a.model, members, scaler, {"seeds": list(a.seeds), "epochs": a.epochs,
                                            "selection": f"mean F1 of N, A, O, ~ on fold {cfg['val_fold']}",
                                            "fs_in": 300})
    n = sum(q.numel() for q in members[0].parameters())
    print(f"Saved {a.model}: {len(members)} members x {n:,} parameters")


if __name__ == "__main__":
    main()
