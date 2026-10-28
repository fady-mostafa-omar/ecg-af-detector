"""External test on long Holter recordings that were never used for training: the 22 MIT-BIH DS2 patients (lead MLII,
360 Hz, 30 min each) and the 23 MIT-BIH AF Database patients with signals (first ECG channel, 250 Hz, about 10 h each).
Every 30-s window (15-s step) is classified and compared with the cardiologists' rhythm annotations; the AF burden
(share of time in AF) is compared per patient. Per-record predictions are cached, so a re-run resumes."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import argparse
import time
from multiprocessing import Pool

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import wfdb

from afsuite.cnn import load_members, make_windows, predict_windows, window_starts
from afsuite.config import load_config
from afsuite.data import CLASSES, NAMES
from afsuite.detect import detect_r_peaks
from afsuite.external import (AFDB_NO_SIGNAL, DS1, DS2, af_fraction, rhythm_intervals, rhythm_seconds,
                              window_metrics)

COLORS = {"N": "#2e9e5b", "A": "#dc2626", "O": "#f39c12", "~": "#9ca3af"}


def read_signal(db, folder, rec):
    if db == "mitdb":
        return wfdb.rdrecord(os.path.join(folder, rec), channel_names=["MLII"]).p_signal[:, 0]
    return wfdb.rdrecord(os.path.join(folder, rec)).p_signal[:, 0]


def prepare_record(args):
    """Worker: (signal, fs) -> window inputs (float16) and window features; the CPU-heavy part."""
    sig, fs = args
    w, f = make_windows(sig, fs, detect_r_peaks(sig, fs))
    return w.astype(np.float16), f


def process(db, folder, recs, cache, members, scaler, gb, device):
    """Classify every window of every record; results saved per record as cache/<db>_<rec>.npz."""
    todo = [r for r in recs if not os.path.exists(os.path.join(cache, f"{db}_{r}.npz"))]
    print(f"{db}: {len(recs) - len(todo)} records cached, {len(todo)} to process", flush=True)
    if not todo:
        return
    t0 = time.time()
    heads = {r: wfdb.rdheader(os.path.join(folder, r)) for r in todo}
    jobs = ((read_signal(db, folder, r), heads[r].fs) for r in todo)
    with Pool(os.cpu_count()) as pool:
        for r, (w, f) in zip(todo, pool.imap(prepare_record, jobs)):
            starts = np.array(window_starts(heads[r].sig_len, heads[r].fs))
            feats = pd.DataFrame(f)
            F = scaler(feats.to_numpy(float))
            pc = np.mean([predict_windows(m, w.astype(np.float32), F, device) for m in members], 0)
            ph = gb["model"].predict(feats[gb["features"]])
            np.savez(os.path.join(cache, f"{db}_{r}.npz"), starts=starts,
                     af=af_fraction(starts, rhythm_intervals(folder, r)), p_cnn=pc, pred_hgb=ph.astype(str))
            print(f"   {db} {r}: {len(starts):,} windows ({time.time() - t0:.0f} s)", flush=True)


def load_rows(db, recs, cache):
    rows = []
    for r in recs:
        d = np.load(os.path.join(cache, f"{db}_{r}.npz"))
        rows.append({"db": db, "record": r, "starts": d["starts"], "af": d["af"],
                     "cnn": np.array(CLASSES)[d["p_cnn"].argmax(1)], "hgb": d["pred_hgb"]})
    return rows


def describe(mitdb_dir, afdb_dir, results):
    """AF and flutter in each MIT-BIH record, and AF share of each AF Database record (annotations only)."""
    rows = []
    for name, recs in (("DS1", DS1), ("DS2", DS2)):
        for r in recs:
            h = wfdb.rdheader(os.path.join(mitdb_dir, r))
            sec = rhythm_seconds(rhythm_intervals(mitdb_dir, r))
            if sec.get("(AFIB", 0) + sec.get("(AFL", 0) > 0:
                rows.append({"split": name, "record": r, "AF min": round(sec.get("(AFIB", 0) / 60, 1),
                             "flutter min": round(sec.get("(AFL", 0) / 60, 1),
                             "record min": round(h.sig_len / h.fs / 60, 1)})
    mit = pd.DataFrame(rows)
    recs = [r.strip() for r in open(os.path.join(afdb_dir, "RECORDS")) if r.strip()]
    rows = []
    for r in recs:
        if r in AFDB_NO_SIGNAL or not os.path.exists(os.path.join(afdb_dir, f"{r}.hea")):
            rows.append({"record": r, "hours": None, "AF %": None, "flutter %": None, "note": "no signal file"})
            continue
        h = wfdb.rdheader(os.path.join(afdb_dir, r))
        total = h.sig_len / h.fs
        sec = rhythm_seconds(rhythm_intervals(afdb_dir, r))
        rows.append({"record": r, "hours": round(total / 3600, 1), "AF %": round(100 * sec.get("(AFIB", 0) / total, 1),
                     "flutter %": round(100 * sec.get("(AFL", 0) / total, 1), "note": ""})
    af = pd.DataFrame(rows)
    mit.to_csv(os.path.join(results, "mitdb_af_episodes.csv"), index=False)
    af.to_csv(os.path.join(results, "afdb_summary.csv"), index=False)
    ok = af["hours"].notna()
    print(f"MIT-BIH records with AF or flutter: {', '.join(mit['record'])} | AF Database: {ok.sum()} records with "
          f"signals, {af.loc[ok, 'hours'].sum():.0f} h")
    return [r for r in recs if r not in AFDB_NO_SIGNAL]


def false_alarms(rows, mitdb_dir, threshold):
    """Per DS2 record: false AF windows of the model, and the share of beats the cardiologists marked as premature
    (ectopic) beats, the usual cause of irregular timing without AF."""
    beats, ectopic = set("NLRBAaJSVrFejnE/fQ"), set("AaJSVE")
    out = []
    for r in rows:
        sym = np.array(wfdb.rdann(os.path.join(mitdb_dir, r["record"]), "atr").symbol)
        sym = sym[np.isin(sym, list(beats))]
        t, p = r["af"] > threshold, r["cnn"] == "A"
        out.append({"record": r["record"], "windows": len(t), "AF windows (cardiologists)": int(t.sum()),
                    "AF windows (model)": int(p.sum()), "false AF windows": int((p & ~t).sum()),
                    "extra beats %": round(100 * np.isin(sym, list(ectopic)).mean(), 1) if len(sym) else np.nan})
    return pd.DataFrame(out)


def timelines_png(rows, burden, path):
    """Cardiologists vs model, window by window, for the DS2 patients with AF and those with false AF alarms."""
    b = burden[burden["database"] == "mitdb"].set_index("record")
    show = [r["record"] for r in rows if b.loc[r["record"], "true AF %"] > 0]
    show += [r["record"] for r in rows if b.loc[r["record"], "true AF %"] == 0 and b.loc[r["record"], "CNN AF %"] > 10]
    by = {r["record"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(11, 0.75 * len(show) + 1.4))
    for k, rec in enumerate(show):
        r = by[rec]
        ends = np.r_[r["starts"][1:], r["starts"][-1] + 30] / 60
        y0 = len(show) - 1 - k
        for i, t in enumerate(r["starts"] / 60):
            ax.add_patch(plt.Rectangle((t, y0 + 0.38), ends[i] - t, 0.3,
                                       color=COLORS["A"] if r["af"][i] > 0.5 else "#e5e7eb", lw=0))
            ax.add_patch(plt.Rectangle((t, y0 + 0.02), ends[i] - t, 0.3, color=COLORS[r["cnn"][i]], lw=0))
        ax.text(-0.4, y0 + 0.35, f"{rec}", ha="right", va="center", fontsize=10, fontweight="bold")
        ax.text(ends[-1] + 0.4, y0 + 0.35, f"AF: cardiologists {b.loc[rec, 'true AF %']:.0f}%, "
                                           f"model {b.loc[rec, 'CNN AF %']:.0f}%", va="center", fontsize=8.5)
    ax.set_xlim(-3, 41)
    ax.set_ylim(-0.1, len(show))
    ax.set_yticks([])
    ax.set_xticks(range(0, 31, 5))
    ax.set_xlabel("time (min)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [Patch(color=COLORS["A"], label="AF (cardiologists: upper bar; model: lower bar)"),
               Patch(color="#e5e7eb", label="not AF (cardiologists)")]
    handles += [Patch(color=COLORS[c], label=f"model: {NAMES[c]}") for c in ("N", "O", "~")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    ax.set_title("Unseen MIT-BIH patients: AF according to the cardiologists (upper bar) and the model (lower bar)",
                 fontsize=11, loc="left")
    fig.tight_layout(rect=(0, 0.7 / fig.get_figheight(), 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    cfg = load_config()
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("mitdb_dir", "afdb_dir", "model", "baseline_model", "results"):
        p.add_argument(f"--{k}", default=cfg[k])
    p.add_argument("--cache", default=cfg["external_cache"])
    p.add_argument("--only", default=None, help="process one record and stop, e.g. mitdb:202 (quick check)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    members, scaler, _ = load_members(a.model)
    members = [m.to(a.device) for m in members]
    gb = joblib.load(a.baseline_model)
    folders = {"mitdb": a.mitdb_dir, "afdb": a.afdb_dir}
    if a.only:
        db, rec = a.only.split(":")
        process(db, folders[db], [rec], a.cache, members, scaler, gb, a.device)
        r = load_rows(db, [rec], a.cache)[0]
        print(f"{db} {rec}: {len(r['starts'])} windows | AF per cardiologists {100 * np.mean(r['af'] > 0.5):.1f}% | "
              f"AF per model {100 * np.mean(r['cnn'] == 'A'):.1f}%")
        return

    af_recs = describe(a.mitdb_dir, a.afdb_dir, a.results)
    process("mitdb", a.mitdb_dir, DS2, a.cache, members, scaler, gb, a.device)
    process("afdb", a.afdb_dir, af_recs, a.cache, members, scaler, gb, a.device)
    mit_rows, afdb_rows = load_rows("mitdb", DS2, a.cache), load_rows("afdb", af_recs, a.cache)

    thr = cfg["af_window_threshold"]
    summary = []
    for name, rows in (("MIT-BIH DS2", mit_rows), ("MIT-BIH AF DB", afdb_rows)):
        for key, label in (("cnn", "CNN ensemble"), ("hgb", "Gradient boosting")):
            m = window_metrics(np.concatenate([r["af"] for r in rows]), np.concatenate([r[key] for r in rows]), thr)
            summary.append({"database": name, "model": label, **m})
    summary = pd.DataFrame(summary).round(1)
    print("\nAF detection on 30-s windows (external data, never used for training):")
    print(summary.to_string(index=False))
    burden = pd.DataFrame([{"database": r["db"], "record": r["record"], "true AF %": 100 * np.mean(r["af"]),
                            "CNN AF %": 100 * np.mean(r["cnn"] == "A"), "GB AF %": 100 * np.mean(r["hgb"] == "A")}
                           for r in mit_rows + afdb_rows]).round(1)
    for db in ("mitdb", "afdb"):
        b = burden[burden["database"] == db]
        print(f"{db}: AF burden per patient (CNN): mean |error| {np.mean(np.abs(b['CNN AF %'] - b['true AF %'])):.1f} "
              f"points | correlation r = {np.corrcoef(b['true AF %'], b['CNN AF %'])[0, 1]:.2f}")

    fig, ax = plt.subplots(figsize=(5, 5))
    for db, c, lab in (("afdb", "#c0392b", f"MIT-BIH AF DB ({len(afdb_rows)} patients)"),
                       ("mitdb", "#2563eb", f"MIT-BIH DS2 ({len(mit_rows)} patients)")):
        b = burden[burden["database"] == db]
        ax.scatter(b["true AF %"], b["CNN AF %"], s=40, color=c, label=lab, alpha=0.8)
    ax.plot([0, 100], [0, 100], color="#999", ls="--", lw=1)
    ax.set(xlabel="AF burden from cardiologists' annotations (%)", ylabel="AF burden from the model (%)",
           xlim=(-3, 103), ylim=(-3, 103), title="AF burden per patient, external data")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.results, "af_burden_external.png"), dpi=150)
    timelines_png(mit_rows, burden, os.path.join(a.results, "af_timelines_mitdb.png"))
    fa = false_alarms(mit_rows, a.mitdb_dir, thr)
    fa.to_csv(os.path.join(a.results, "mitdb_false_alarms.csv"), index=False)
    top = fa.sort_values("false AF windows", ascending=False).head(4)
    print(f"MIT-BIH DS2: {fa['false AF windows'].sum()} false AF windows; most in records "
          + ", ".join(f"{r['record']} ({r['false AF windows']}, extra beats {r['extra beats %']:.0f}%)"
                      for _, r in top.iterrows()))
    summary.to_csv(os.path.join(a.results, "external_windows.csv"), index=False)
    burden.to_csv(os.path.join(a.results, "external_burden.csv"), index=False)


if __name__ == "__main__":
    main()
