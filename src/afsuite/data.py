"""CinC 2017 data access: the compact npz written in Step 2.1 and ragged arrays of R peaks."""
import numpy as np

CLASSES = ["N", "A", "O", "~"]
NAMES = {"N": "Normal", "A": "AF", "O": "Other", "~": "Noisy"}


def load_cinc(npz_path):
    """Return (signals in mV, record names, labels, sampling rate)."""
    d = np.load(npz_path, allow_pickle=True)
    samples, off, gain = d["samples"], d["offsets"], d["gain"]      # load each array once
    sigs = [samples[off[i]:off[i + 1]] / gain[i] for i in range(len(off) - 1)]
    return sigs, d["record"].astype(str), d["label"].astype(str), int(d["fs"][0])


def save_ragged(path, arrays):
    lens = np.array([len(a) for a in arrays])
    np.savez_compressed(path, values=np.concatenate(arrays).astype(np.int32), offsets=np.r_[0, np.cumsum(lens)])


def load_ragged(path):
    d = np.load(path)
    v, off = d["values"], d["offsets"]
    return [v[off[i]:off[i + 1]] for i in range(len(off) - 1)]
