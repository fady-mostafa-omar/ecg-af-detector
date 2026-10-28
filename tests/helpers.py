"""Synthetic single-lead ECG for the tests (no real data needed)."""
import numpy as np


def synthetic_ecg(seconds=60, fs=300, rr=0.8, jitter=0.0, invert=False, seed=0):
    """Gaussian QRS complexes and T waves plus noise; returns (signal in mV, true R-peak sample indices)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    beats, x = [], 0.5
    while x < seconds - 0.5:
        beats.append(x)
        x += rr * (1 + jitter * rng.uniform(-1, 1))
    sig = rng.normal(0, 0.02, len(t))
    for b in beats:
        sig += 1.0 * np.exp(-0.5 * ((t - b) / 0.010) ** 2) + 0.25 * np.exp(-0.5 * ((t - b - 0.25) / 0.040) ** 2)
    return (-sig if invert else sig), np.round(np.array(beats) * fs).astype(int)
