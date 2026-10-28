import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from afsuite.cnn import (AFNet, FeatureScaler, L, load_members, make_windows, per_recording, predict_windows,
                         save_members, window_starts)
from afsuite.detect import detect_r_peaks
from helpers import synthetic_ecg


def test_window_starts_cover_the_recording():
    assert window_starts(20 * 300, 300) == [0.0]
    assert np.allclose(window_starts(60 * 300, 300), [0, 15, 30])
    for seconds in (31, 45.5, 47, 61.5, 600.2, 1800):
        s = window_starts(int(seconds * 300), 300)
        assert s[0] == 0 and np.all(np.diff(s) <= 15 + 1e-9) and s[-1] + 30 >= seconds - 1


def test_make_windows_shapes_at_any_sampling_rate():
    for fs in (300, 360, 250):
        sig, _ = synthetic_ecg(seconds=70, fs=fs)
        waves, feats = make_windows(sig, fs, detect_r_peaks(sig, fs))
        assert waves.shape == (len(window_starts(len(sig), fs)), L) and waves.dtype == np.float32
        assert len(feats) == len(waves) and all(len(f) == 35 for f in feats)


def test_afnet_output_and_size():
    torch.manual_seed(0)
    m = AFNet(35).eval()
    with torch.no_grad():
        out = m(torch.randn(3, 1, L), torch.randn(3, 35))
    assert out.shape == (3, 4)
    assert sum(p.numel() for p in m.parameters()) == 820_516


def test_feature_scaler_handles_missing_and_extreme_values():
    F = np.array([[1.0, 10.0], [3.0, 20.0], [2.0, np.nan]])
    sc = FeatureScaler().fit(F)
    out = sc(np.array([[np.nan, 1e9], [2.0, 15.0]]))
    assert out.dtype == np.float32 and out[0, 0] == 0 and out[0, 1] == 5 and np.allclose(out[1], 0, atol=1e-5)


def test_per_recording_averages_window_probabilities():
    p = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
    P = per_recording(p, np.array([0, 0, 2]), 3)
    assert np.allclose(P[0], [0.5, 0.5, 0, 0]) and np.allclose(P[1], 0) and np.allclose(P[2], [0, 0, 1, 0])


def test_ensemble_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    members = [AFNet(35).eval(), AFNet(35).eval()]
    sc = FeatureScaler().fit(np.random.default_rng(0).normal(size=(50, 35)))
    path = str(tmp_path / "ens.pt")
    save_members(path, members, sc, {"seeds": [0, 1]})
    loaded, sc2, ck = load_members(path)
    w = np.random.default_rng(1).normal(size=(2, L)).astype(np.float32)
    f = sc2(np.random.default_rng(2).normal(size=(2, 35)))
    assert ck["seeds"] == [0, 1] and len(loaded) == 2 and np.allclose(sc2.mu, sc.mu)
    for a, b in zip(members, loaded):
        assert np.allclose(predict_windows(a, w, f, "cpu"), predict_windows(b, w, f, "cpu"), atol=1e-6)
