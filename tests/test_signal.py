import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afsuite.data import CLASSES
from afsuite.detect import detect_r_peaks
from afsuite.features import BEAT_KEYS, RR_KEYS, recording_features, rr_features, sample_entropy
from afsuite.metrics import challenge_score, confusion
from helpers import synthetic_ecg


def test_challenge_score_is_mean_f1_of_normal_af_other():
    y_true = np.array(list("NNNNAAOO~~"))
    y_pred = np.array(list("NNNAAAON~N"))
    score, f1 = challenge_score(y_true, y_pred)
    # F1 = 2 x correct / (reference count + predicted count)
    assert np.isclose(f1["N"], 2 * 3 / (4 + 5))
    assert np.isclose(f1["A"], 2 * 2 / (2 + 3))
    assert np.isclose(f1["O"], 2 * 1 / (2 + 1))
    assert np.isclose(f1["~"], 2 * 1 / (2 + 1))
    assert np.isclose(score, np.mean([f1["N"], f1["A"], f1["O"]]))      # Noisy is not averaged
    assert challenge_score(y_true, y_true)[0] == 1.0
    cm = confusion(y_true, y_pred)
    assert list(cm.index) == CLASSES and cm.to_numpy().sum() == len(y_true) and cm.loc["O", "N"] == 1


def test_r_peaks_found_for_both_polarities():
    for invert in (False, True):
        sig, truth = synthetic_ecg(seconds=60, invert=invert)
        peaks = detect_r_peaks(sig, 300)
        d = np.abs(peaks[:, None] - truth[None, :]).min(0)
        assert np.mean(d <= 0.05 * 300) > 0.95, f"invert={invert}: {len(peaks)} peaks for {len(truth)} beats"
        assert len(peaks) <= len(truth) + 2


def test_rr_features_separate_regular_from_irregular():
    reg = rr_features(np.full(40, 0.8))
    rng = np.random.default_rng(0)
    irr = rr_features(rng.uniform(0.4, 1.2, 40))
    assert reg["irregular_frac"] == 0 and reg["cv"] < 1e-9 and reg["pnn50"] == 0
    assert irr["irregular_frac"] > 0.3 and irr["cv"] > 0.15 and irr["rmssd"] > reg["rmssd"]
    assert sample_entropy(rng.uniform(0.4, 1.2, 300))[0] > 1.5        # random intervals: high entropy
    assert sample_entropy(np.tile([0.7, 0.9], 150))[0] < 0.1          # repeating pattern: low entropy
    assert np.isnan(sample_entropy(np.ones(30))[0])                   # no variability -> undefined
    assert np.isnan(rr_features(np.array([0.8, 0.8]))["sdnn"])      # too few intervals


def test_recording_features_has_35_named_values():
    sig, _ = synthetic_ecg(seconds=30)
    f = recording_features(sig, 300, detect_r_peaks(sig, 300))
    assert len(f) == 35 and set(RR_KEYS) <= set(f) and set(BEAT_KEYS) <= set(f)
    assert abs(f["hr_mean"] - 75) < 2 and f["corr_mean"] > 0.95 and abs(f["duration"] - 30) < 1e-9
