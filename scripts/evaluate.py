"""Score the shipped models on the locked CinC 2017 test set: each CNN seed alone, the 5-seed ensemble and the
gradient-boosting baseline; paired bootstrap of the difference; AF detection per recording; model size and speed."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from afsuite.cnn import load_members, make_windows, per_recording, predict_windows
from afsuite.config import load_config
from afsuite.data import CLASSES, NAMES, load_cinc
from afsuite.detect import detect_r_peaks
from afsuite.metrics import challenge_score, confusion


def af_stats(y_true, y_pred):
    t, p = np.asarray(y_true) == "A", np.asarray(y_pred) == "A"
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    return {"AF recordings": int(t.sum()), "AF found": int(tp), "false AF": int(fp),
            "Se %": round(100 * tp / max(tp + fn, 1), 1), "PPV %": round(100 * tp / max(tp + fp, 1), 1),
            "Specificity %": round(100 * tn / max(tn + fp, 1), 1)}


def confusion_png(cms, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(4.6 * len(cms), 4.2))
    for ax, (title, cm) in zip(np.atleast_1d(axes), cms.items()):
        pct = 100 * cm.to_numpy() / np.maximum(cm.to_numpy().sum(1, keepdims=True), 1)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{pct[i, j]:.0f}%\n({cm.iloc[i, j]})", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 55 else "#111")
        names = [NAMES[c] for c in CLASSES]
        ax.set_xticks(range(len(CLASSES)), names)
        ax.set_yticks(range(len(CLASSES)), names)
        ax.set_xlabel("predicted")
        ax.set_ylabel("expert label")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Locked test set: what each expert label was predicted as (% of the row, count)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("dataset", "windows", "window_features", "features", "split", "model", "baseline_model", "results"):
        p.add_argument(f"--{k}", default=cfg[k])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    os.makedirs(a.results, exist_ok=True)

    split = pd.read_csv(a.split, dtype={"record": str, "label": str})
    labels = split["label"].to_numpy()
    test = (split["split"] == "test").to_numpy()
    test_recs = np.where(test)[0]
    yt = labels[test_recs]
    z = np.load(a.windows)
    owner = z["owner"]
    te_idx = np.where(test[owner])[0]
    waves = z["waves"][te_idx].astype(np.float32)
    members, scaler, ck = load_members(a.model)
    feats = scaler(pd.read_csv(a.window_features).to_numpy(float)[te_idx])
    print(f"Test set: {len(test_recs):,} recordings, {len(te_idx):,} windows | {len(members)} members | {a.device}")

    # every member alone, then the ensemble (average of the members' window probabilities)
    probs, rows = [], []
    for s, m in zip(ck.get("seeds", range(len(members))), members):
        pw = predict_windows(m.to(a.device), waves, feats, a.device)
        m.cpu()
        probs.append(pw)
        pred = np.array(CLASSES)[per_recording(pw, owner[te_idx], len(labels))[test_recs].argmax(1)]
        sc, f1 = challenge_score(yt, pred)
        row = {"seed": s, "official score": sc}
        row.update({f"F1 {NAMES[c]}": f1[c] for c in CLASSES})
        rows.append(row)
    seeds = pd.DataFrame(rows)
    pred_cnn = np.array(CLASSES)[per_recording(np.mean(probs, 0), owner[te_idx], len(labels))[test_recs].argmax(1)]
    score, f1 = challenge_score(yt, pred_cnn)

    gb = joblib.load(a.baseline_model)
    F = pd.read_csv(a.features)
    pred_gb = gb["model"].predict(F.loc[test, gb["features"]])
    gscore, gf1 = challenge_score(yt, pred_gb)

    cols = ["official score"] + [f"F1 {NAMES[c]}" for c in CLASSES]
    pm = lambda c: f"{seeds[c].mean():.3f} ± {seeds[c].std(ddof=1):.3f}"
    table = pd.DataFrame({"Gradient boosting": [f"{gscore:.3f}"] + [f"{gf1[c]:.3f}" for c in CLASSES],
                          f"CNN, single (mean ± SD, {len(members)} seeds)": [pm(c) for c in cols],
                          f"CNN, {len(members)}-seed ensemble": [f"{score:.3f}"] + [f"{f1[c]:.3f}" for c in CLASSES]},
                         index=cols)
    print("\nOfficial score = mean F1 of Normal, AF and Other:")
    print(table.to_string())
    cm = confusion(yt, pred_cnn)
    print("\nEnsemble confusion matrix (rows: expert label):")
    print(cm.to_string())

    # paired bootstrap of (ensemble - gradient boosting) over the test recordings
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    diffs = []
    for _ in range(cfg["bootstrap"]):
        i = rng.integers(0, len(yt), len(yt))
        diffs.append(challenge_score(yt[i], pred_cnn[i])[0] - challenge_score(yt[i], pred_gb[i])[0])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    d0 = score - gscore
    verdict = ("ensemble better beyond chance" if lo > 0 else "ensemble worse beyond chance" if hi < 0
               else "not clearly different (the interval includes 0)")
    print(f"\nEnsemble minus gradient boosting: {d0:+.3f} (95% bootstrap interval {lo:+.3f} to {hi:+.3f}) -> {verdict}")
    af = pd.DataFrame([{"model": "Gradient boosting", **af_stats(yt, pred_gb)},
                       {"model": "CNN ensemble", **af_stats(yt, pred_cnn)}])
    print("\nAF detection per recording:")
    print(af.to_string(index=False))

    # size and speed on the CPU (one thread per core)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    n_par = sum(q.numel() for q in members[0].parameters())
    for m in members:
        m.cpu()
    k = min(64, len(waves))
    for m in members:
        predict_windows(m, waves[:8], feats[:8], "cpu")                     # warm-up
    t0 = time.time()
    for m in members:
        predict_windows(m, waves[:k], feats[:k], "cpu")
    ms_window = 1000 * (time.time() - t0) / k
    sigs, _, _, fs = load_cinc(a.dataset)
    long = np.concatenate([sigs[i] for i in test_recs])[: int(30 * 60 * fs)]  # 30 min of test recordings, joined
    t0 = time.time()
    peaks = detect_r_peaks(long, fs)
    w, wf = make_windows(long, fs, peaks)
    wf = scaler(pd.DataFrame(wf).to_numpy(float))
    for m in members:
        predict_windows(m, w, wf, "cpu")
    s30 = time.time() - t0
    info = {"members": len(members), "parameters_per_member": n_par, "parameters_total": n_par * len(members),
            "model_file_MB": round(os.path.getsize(a.model) / 1e6, 1), "cpu_threads": torch.get_num_threads(),
            "ms_per_30s_window_cpu_network_only": round(ms_window, 1),
            "s_per_30min_ecg_cpu_full_pipeline": round(s30, 1), "windows_in_30min": len(w)}
    print(f"\nSize and speed (CPU, {info['cpu_threads']} threads): {n_par:,} parameters per member; "
          f"ensemble {ms_window:.0f} ms per 30-s window (network only); "
          f"30 min of ECG analyzed end to end in {s30:.1f} s ({len(w)} windows)")

    seeds.round(4).to_csv(os.path.join(a.results, "cnn_seeds_test.csv"), index=False)
    table.to_csv(os.path.join(a.results, "cnn_vs_baseline_test.csv"))
    cm.to_csv(os.path.join(a.results, "cnn_confusion.csv"))
    pd.DataFrame([{"diff": d0, "ci_low": lo, "ci_high": hi}]).to_csv(
        os.path.join(a.results, "bootstrap_cnn_vs_gb.csv"), index=False)
    af.to_csv(os.path.join(a.results, "af_detection_test.csv"), index=False)
    confusion_png({"Gradient boosting": confusion(yt, pred_gb), f"CNN, {len(members)}-seed ensemble": cm},
                  os.path.join(a.results, "confusion_test.png"))
    with open(os.path.join(a.results, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
