import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
sys.path.insert(0, os.path.dirname(__file__))
import app as A
from helpers import synthetic_ecg


def test_upload_helpers(tmp_path):
    sig, _ = synthetic_ecg(seconds=20)
    f = tmp_path / "ecg.csv"
    np.savetxt(f, np.c_[0.3 * sig, sig], delimiter=",", header="I,II", comments="")
    names, header = A.read_columns(str(f))
    assert header and names == ["I", "II"] and A.default_column(names) == "II"
    assert np.allclose(A.read_signal(str(f), "II"), sig)
    raw = -512 * sig + 2048                                   # ADC counts, upside down
    fixed, notes = A.auto_adjust(raw, 300)
    assert len(notes) == 2 and np.corrcoef(fixed, sig)[0, 1] > 0.99
    assert np.percentile(np.abs(A.bandpass(fixed, 300)), 99.5) < 5


def test_runs_merge_consecutive_windows():
    res = {"cls": np.array(list("NAANA")), "starts": np.array([0, 15, 30, 45, 60.0]),
           "ends": np.array([15, 30, 45, 60, 90.0])}
    assert A.runs(res, "A") == [(15, 45), (60, 90)]
    assert A.runs(res, "O") == []
