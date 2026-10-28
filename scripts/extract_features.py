"""R peaks, recording features (for the gradient-boosting baseline) and 30-s windows with their features (for the
CNN), for every CinC 2017 recording. Uses all CPU cores. Existing output files are reused unless --force is given."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from afsuite.cnn import windows_one
from afsuite.config import load_config
from afsuite.data import CLASSES, NAMES, load_cinc, load_ragged, save_ragged
from afsuite.detect import detect_one
from afsuite.features import features_one


def run_pool(fn, jobs, chunksize, label):
    out, t0 = [], time.time()
    with Pool(os.cpu_count()) as pool:
        for i, r in enumerate(pool.imap(fn, jobs, chunksize=chunksize), 1):
            out.append(r)
            if i % 2000 == 0:
                print(f"   {label}: {i:,}/{len(jobs):,} recordings ({time.time() - t0:.0f} s)", flush=True)
    print(f"{label}: {len(jobs):,} recordings in {time.time() - t0:.0f} s", flush=True)
    return out


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("dataset", "peaks", "features", "windows", "window_features"):
        p.add_argument(f"--{k}", default=cfg[k])
    p.add_argument("--limit", type=int, default=None, help="only the first N recordings (quick check)")
    p.add_argument("--force", action="store_true", help="recompute even if the output files exist")
    a = p.parse_args()
    sigs, records, labels, fs = load_cinc(a.dataset)
    if a.limit:
        sigs, labels = sigs[:a.limit], labels[:a.limit]
    for path in (a.peaks, a.features, a.windows, a.window_features):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if os.path.exists(a.peaks) and not a.force:
        peaks = load_ragged(a.peaks)
        print(f"R peaks: reused {a.peaks}")
    else:
        peaks = run_pool(detect_one, [(s, fs) for s in sigs], 40, "R peaks")
        save_ragged(a.peaks, peaks)
    assert len(peaks) == len(sigs), "cached R peaks do not match the recordings; use --force"
    n_beats = np.array([len(q) for q in peaks])
    print("   beats per recording (median):", {NAMES[c]: int(np.median(n_beats[labels == c]))
                                                for c in CLASSES if (labels == c).any()},
          "| recordings with fewer than 5 beats:", int((n_beats < 5).sum()))

    if os.path.exists(a.features) and not a.force:
        print(f"Recording features: reused {a.features}")
    else:
        F = pd.DataFrame(run_pool(features_one, [(s, fs, q) for s, q in zip(sigs, peaks)], 40, "Recording features"))
        F.to_csv(a.features, index=False)
        print(f"   {F.shape[1]} features -> {a.features}")

    if os.path.exists(a.windows) and os.path.exists(a.window_features) and not a.force:
        print(f"Windows: reused {a.windows}")
    else:
        res = run_pool(windows_one, [(s, fs, q) for s, q in zip(sigs, peaks)], 20, "30-s windows")
        waves = np.concatenate([w.astype(np.float16) for w, _ in res])
        owner = np.concatenate([np.full(len(w), i) for i, (w, _) in enumerate(res)])
        np.savez(a.windows, waves=waves, owner=owner)
        pd.DataFrame([f for _, fl in res for f in fl]).to_csv(a.window_features, index=False)
        print(f"   {len(owner):,} windows of {waves.shape[1]} samples -> {a.windows}")


if __name__ == "__main__":
    main()
