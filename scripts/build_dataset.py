"""Read the CinC 2017 training set into one compact file (int16 samples) and create or check the locked split."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from afsuite.config import load_config
from afsuite.data import CLASSES, NAMES
from afsuite.splits import make_split, recording_hashes


def find_labels(cinc_dir, name):
    """The label file `name` (anywhere outside validation/), else training/REFERENCE.csv."""
    cands = sorted((p for p in glob.glob(os.path.join(cinc_dir, "**", name), recursive=True)
                    if "validation" not in os.path.relpath(p, cinc_dir)), key=len)
    cands.append(os.path.join(cinc_dir, "training", "REFERENCE.csv"))
    path = next((p for p in cands if os.path.exists(p)), None)
    assert path, f"No label file ({name}) in {cinc_dir}; run scripts/download_data.sh"
    return pd.read_csv(path, header=None, names=["record", "label"]), path


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cinc_dir", default=cfg["cinc_dir"])
    p.add_argument("--dataset", default=cfg["dataset"])
    p.add_argument("--split", default=cfg["split"])
    p.add_argument("--results", default=cfg["results"])
    a = p.parse_args()

    ref, used = find_labels(a.cinc_dir, cfg["labels"])
    heads = glob.glob(os.path.join(a.cinc_dir, "training", "**", "*.hea"), recursive=True)
    path_of = {os.path.basename(h)[:-4]: h[:-4] for h in heads}
    missing = [r for r in ref["record"] if r not in path_of]
    assert not missing, f"{len(missing)} labeled recordings not found, e.g. {missing[:3]}; run scripts/download_data.sh"
    sigs, gains, fss = [], [], []
    for rec in ref["record"]:
        x = wfdb.rdrecord(path_of[rec], physical=False)
        sigs.append(x.d_signal[:, 0].astype(np.int16))
        gains.append(x.adc_gain[0])
        fss.append(x.fs)
    lens = np.array([len(s) for s in sigs])
    offsets = np.r_[0, np.cumsum(lens)]
    samples = np.concatenate(sigs)
    os.makedirs(os.path.dirname(os.path.abspath(a.dataset)), exist_ok=True)
    np.savez_compressed(a.dataset, record=ref["record"].to_numpy(), label=ref["label"].to_numpy(), samples=samples,
                        offsets=offsets, gain=np.array(gains), fs=np.array(fss))
    print(f"{len(ref):,} recordings (labels: {os.path.relpath(used, a.cinc_dir)}) -> {a.dataset} "
          f"({os.path.getsize(a.dataset) / 1e6:.0f} MB) | sampling rates {sorted(set(fss))} Hz")

    # recordings per class and their length
    dur = lens / np.array(fss)
    info = pd.DataFrame({"label": ref["label"].map(NAMES), "seconds": dur})
    table = info.groupby("label")["seconds"].describe()[["count", "min", "50%", "max"]].round(1)
    table = table.reindex([NAMES[c] for c in CLASSES]).rename(columns={"50%": "median"})
    table["share %"] = (100 * table["count"] / table["count"].sum()).round(1)
    print(table.to_string())
    os.makedirs(a.results, exist_ok=True)
    table.to_csv(os.path.join(a.results, "cinc2017_summary.csv"))

    # one 10-s example of each class
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    rng = np.random.default_rng(0)
    for ax, code in zip(axes, CLASSES):
        i = rng.choice(np.where((ref["label"] == code).to_numpy() & (dur >= 12))[0])
        x = sigs[i][: int(10 * fss[i])] / gains[i]
        ax.plot(np.arange(len(x)) / fss[i], x, color="#111", lw=0.8)
        ax.set_ylabel("mV")
        ax.set_title(f"{NAMES[code]}  ({ref['record'][i]})", loc="left", fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "examples_per_class.png"), dpi=130)

    # the locked split: re-derived from the data and compared with the shipped file
    hashes = recording_hashes(samples, offsets)
    n_dup = int(pd.Series(hashes).duplicated(keep=False).sum())
    split = make_split(ref["record"].to_numpy(), ref["label"].to_numpy(), hashes, cfg["test_size"], cfg["split_seed"],
                       cfg["n_folds"], cfg["fold_seed"])
    if os.path.exists(a.split):
        locked = pd.read_csv(a.split, dtype={"record": str, "label": str})
        assert (locked["record"].to_numpy() == split["record"].to_numpy()).all(), "split file does not match the data"
        assert (locked["label"].to_numpy() == split["label"].to_numpy()).all(), "split labels do not match the data"
        same = all((locked[c].astype(str).to_numpy() == split[c].astype(str).to_numpy()).all() for c in split.columns)
        print(f"Locked split {a.split}: kept unchanged; re-derived from the data: "
              + ("identical ✅" if same else "DIFFERENT (for example another scikit-learn version); the locked file is used"))
        split = locked
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.split)), exist_ok=True)
        split.to_csv(a.split, index=False)
        print(f"Split created: {a.split}")
    print(f"Recordings with an exact duplicate: {n_dup} (kept on the same side of the split)")
    tab = pd.crosstab(split["split"], split["label"]).reindex(columns=CLASSES).copy()
    tab["total"] = tab.sum(axis=1)
    print(tab.to_string())


if __name__ == "__main__":
    main()
