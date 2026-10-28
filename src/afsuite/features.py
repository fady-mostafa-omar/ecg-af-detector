"""Hand-crafted features of one recording:
RR irregularity (the core signature of AF), average beat shape (P wave, QRS width), and signal quality."""
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .detect import bandpass

RR_KEYS = ["n_rr", "hr_mean", "rr_median", "rr_min", "rr_max", "sdnn", "cv", "rmssd", "nrmssd", "pnn20", "pnn50",
           "irregular_frac", "mad_drr_norm", "sd1", "sd2", "sd1_sd2", "tpr", "rr_entropy", "sampen", "cosen"]
BEAT_KEYS = ["corr_mean", "corr_min", "corr_low_frac", "r_amp_med", "r_amp_cv", "qrs_width", "p_ratio", "t_ratio"]


def sample_entropy(x, m=2, r=0.2):
    x = np.asarray(x, dtype=float)
    tol = r * np.std(x)
    if len(x) < m + 3 or tol == 0:
        return np.nan, tol
    n = len(x) - m
    def pairs(mm):
        emb = np.lib.stride_tricks.sliding_window_view(x, mm)[:n]
        d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
        return (np.sum(d <= tol) - n) / 2
    b, a = pairs(m), pairs(m + 1)
    return (-np.log(a / b) if a > 0 and b > 0 else np.nan), tol


def rr_features(rr):
    out = dict.fromkeys(RR_KEYS, np.nan)
    out["n_rr"] = len(rr)
    if len(rr) < 4:
        return out
    d = np.diff(rr)
    mean = rr.mean()
    sdnn, rmssd = rr.std(), np.sqrt(np.mean(d ** 2))
    sd1 = np.std(d) / np.sqrt(2)
    sd2 = np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0))
    turns = np.sum((rr[1:-1] - rr[:-2]) * (rr[1:-1] - rr[2:]) > 0)
    hist, _ = np.histogram(np.clip(rr, 0.2, 2.0), bins=16, range=(0.2, 2.0))
    p = hist[hist > 0] / hist.sum()
    se, tol = sample_entropy(rr)
    out.update(hr_mean=60 / mean, rr_median=np.median(rr), rr_min=rr.min(), rr_max=rr.max(), sdnn=sdnn,
               cv=sdnn / mean, rmssd=rmssd, nrmssd=rmssd / mean, pnn20=np.mean(np.abs(d) > 0.02),
               pnn50=np.mean(np.abs(d) > 0.05), irregular_frac=np.mean(np.abs(d) / rr[:-1] > 0.15),
               mad_drr_norm=np.median(np.abs(d)) / np.median(rr), sd1=sd1, sd2=sd2,
               sd1_sd2=sd1 / sd2 if sd2 > 0 else np.nan, tpr=turns / ((2 * len(rr) - 4) / 3),
               rr_entropy=-np.sum(p * np.log(p)) / np.log(16), sampen=se,
               cosen=se + np.log(2 * tol / mean) if np.isfinite(se) and tol > 0 else np.nan)
    return out


def beat_features(filt, peaks, fs):
    out = dict.fromkeys(BEAT_KEYS, np.nan)
    pre, post = int(0.25 * fs), int(0.45 * fs)
    keep = peaks[(peaks >= pre) & (peaks + post < len(filt))]
    if len(keep) < 3:
        return out
    W = filt[keep[:, None] + np.arange(-pre, post)]
    tmpl = np.median(W, 0)
    peak = max(abs(tmpl[pre]), 1e-6)
    corr = np.array([np.corrcoef(w, tmpl)[0, 1] for w in W])
    amps = np.abs(W[:, pre])
    q = int(0.07 * fs)
    p_win = tmpl[: pre - int(0.08 * fs)]
    t_win = tmpl[pre + int(0.10 * fs):]
    out.update(corr_mean=np.nanmean(corr), corr_min=np.nanmin(corr), corr_low_frac=np.mean(corr < 0.8),
               r_amp_med=np.median(amps), r_amp_cv=amps.std() / max(amps.mean(), 1e-6),
               qrs_width=np.sum(np.abs(tmpl[pre - q:pre + q]) > 0.5 * peak) / fs,
               p_ratio=np.ptp(p_win) / peak, t_ratio=np.ptp(t_win) / peak)
    return out


def signal_features(sig, filt, fs):
    f, pxx = welch(sig - np.mean(sig), fs=fs, nperseg=min(len(sig), 4 * fs))
    tot = pxx.sum() + 1e-12
    return {"duration": len(sig) / fs, "kurtosis": kurtosis(filt), "skew": skew(filt),
            "pow_lt1": pxx[f < 1].sum() / tot, "pow_5_15": pxx[(f >= 5) & (f <= 15)].sum() / tot,
            "pow_gt40": pxx[f > 40].sum() / tot}


def recording_features(sig, fs, peaks):
    filt = bandpass(sig, fs)
    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.2) & (rr < 3.0)]          # drop physiologically impossible intervals (detection errors)
    out = {"beats_per_s": len(peaks) / (len(sig) / fs)}
    out.update(rr_features(rr))
    out.update(beat_features(filt, peaks, fs))
    out.update(signal_features(sig, filt, fs))
    return out


def features_one(args):
    """(signal, fs, peaks) -> feature dict; a picklable wrapper for multiprocessing."""
    return recording_features(*args)
