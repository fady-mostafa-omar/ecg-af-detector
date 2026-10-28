"""External test data: rhythm annotations of the MIT-BIH Arrhythmia Database and the MIT-BIH AF Database."""
import os

import numpy as np
import wfdb

# Inter-patient split of MIT-BIH (de Chazal et al., 2004), as in ecg-arrhythmia-detector. Only DS2 is used for testing.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]
AFDB_NO_SIGNAL = {"00735", "03665"}          # annotation-only records (no signal file on PhysioNet)


def rhythm_intervals(folder, rec):
    """[(start s, end s, rhythm)] from the '(AFIB', '(N', ... rhythm annotations; each lasts until the next one."""
    h = wfdb.rdheader(os.path.join(folder, rec))
    a = wfdb.rdann(os.path.join(folder, rec), "atr")
    marks = [(s, note.strip("\x00 ")) for s, note in zip(a.sample, a.aux_note) if note and note.startswith("(")]
    return [(s / h.fs, (marks[j + 1][0] if j + 1 < len(marks) else h.sig_len) / h.fs, nm)
            for j, (s, nm) in enumerate(marks)]


def rhythm_seconds(intervals):
    """Seconds spent in each rhythm."""
    out = {}
    for s, e, nm in intervals:
        out[nm] = out.get(nm, 0.0) + e - s
    return out


def af_fraction(starts, intervals, win=30.0):
    """Share of each window [t, t + win) annotated as AF ('(AFIB')."""
    out = []
    for t0 in starts:
        t1 = t0 + win
        out.append(sum(max(0.0, min(t1, e) - max(t0, s)) for s, e, nm in intervals if nm == "(AFIB") / win)
    return np.array(out)


def window_metrics(af_frac, pred, threshold=0.5):
    """AF detection per window. A window is AF if more than `threshold` of it is annotated AF; flutter and every other
    rhythm count as non-AF, and a window the model calls Noisy counts as non-AF (reported separately)."""
    t = np.asarray(af_frac) > threshold
    p = np.asarray(pred) == "A"
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    se, ppv, sp = tp / max(tp + fn, 1), tp / max(tp + fp, 1), tn / max(tn + fp, 1)
    return {"windows": len(t), "AF windows": int(t.sum()), "Se %": 100 * se, "PPV %": 100 * ppv,
            "Specificity %": 100 * sp, "F1 %": 100 * 2 * se * ppv / max(se + ppv, 1e-9),
            "called Noisy %": 100 * np.mean(np.asarray(pred) == "~")}
