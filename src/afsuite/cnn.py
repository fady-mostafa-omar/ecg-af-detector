"""Hybrid model: a residual 1D-CNN on 30-s ECG windows (150 Hz), fused with the hand-crafted window features.
A recording's prediction is the average of its windows' probabilities."""
import time
from math import gcd

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.signal import resample_poly

from .data import CLASSES
from .detect import bandpass
from .features import recording_features
from .metrics import challenge_score

FS_NET, WIN, STEP = 150, 30, 15            # network input rate (Hz), window length and step (s)
L = FS_NET * WIN


def prepare(sig, fs):
    """Band-pass 0.5-40 Hz, resample to 150 Hz, scale so that R peaks are about 1 (polarity kept)."""
    x = bandpass(sig, fs)
    g = gcd(FS_NET, int(fs))
    x = resample_poly(x, FS_NET // g, int(fs) // g) if int(fs) != FS_NET else x
    return (x / (np.percentile(np.abs(x), 99) + 1e-6)).astype(np.float32)


def window_starts(n_samples, fs):
    """Start times (s) of 30-s windows with a 15-s step covering the recording (one window if shorter)."""
    dur = n_samples / fs
    if dur <= WIN:
        return [0.0]
    starts = list(np.arange(0, dur - WIN + 1e-9, STEP))
    if dur - WIN - starts[-1] > 1:
        starts.append(dur - WIN)
    return starts


def make_windows(sig, fs, peaks):
    """Network inputs (n, 4500) and hand-crafted features (list of dicts) for every window of one recording."""
    x = prepare(sig, fs)
    waves, feats = [], []
    for t0 in window_starts(len(sig), fs):
        a = int(round(t0 * FS_NET))
        w = x[a:a + L]
        waves.append(np.pad(w, (0, L - len(w))))
        lo, hi = int(t0 * fs), int((t0 + WIN) * fs)
        p = peaks[(peaks >= lo) & (peaks < hi)] - lo
        feats.append(recording_features(sig[lo:hi], fs, p))
    return np.stack(waves), feats


def windows_one(args):
    """(signal, fs, peaks) -> (waves, feature dicts); a picklable wrapper for multiprocessing."""
    return make_windows(*args)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, k=7, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x):
        return Fn.relu(self.b2(self.c2(Fn.relu(self.b1(self.c1(x))))) + self.skip(x))


class AFNet(nn.Module):
    def __init__(self, n_feat, n_classes=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 15, 2, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        plan = [(32, 32, 2), (32, 64, 2), (64, 64, 2), (64, 128, 2), (128, 128, 2), (128, 128, 2)]
        self.blocks = nn.Sequential(*[ResBlock(a, b, 7, s) for a, b, s in plan])
        self.feat = nn.Sequential(nn.Linear(n_feat, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128 * 2 + 64, 64), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, wave, feats):
        h = self.blocks(self.stem(wave))
        return self.head(torch.cat([h.mean(-1), h.amax(-1), self.feat(feats)], 1))


class FeatureScaler:
    """Standardize with training statistics; missing values become 0 (= the training mean); clip to +-5."""
    def fit(self, F):
        self.mu, self.sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-6
        return self

    def __call__(self, F):
        return np.clip(np.nan_to_num((F - self.mu) / self.sd), -5, 5).astype(np.float32)


@torch.no_grad()
def predict_windows(model, waves, feats, device, batch=256):
    model.eval()
    out = []
    for i in range(0, len(waves), batch):
        w = torch.as_tensor(waves[i:i + batch], device=device).unsqueeze(1)
        f = torch.as_tensor(feats[i:i + batch], device=device)
        out.append(torch.softmax(model(w, f), 1).float().cpu())
    return torch.cat(out).numpy()


def per_recording(prob_windows, owner, n_records):
    """Average window probabilities per recording."""
    P = np.zeros((n_records, prob_windows.shape[1]))
    np.add.at(P, owner, prob_windows)
    return P / np.maximum(np.bincount(owner, minlength=n_records), 1)[:, None]


def select_score(y_true, y_pred):
    """Epoch selection: mean F1 of all four classes, so that Noisy (not in the official score) is not neglected."""
    _, f1 = challenge_score(y_true, y_pred)
    return float(np.nanmean([f1[c] for c in CLASSES]))


def train(waves, feats, y_win, train_idx, val_idx, owner, y_rec, val_recs, epochs=30, batch=64, lr=1e-3,
          seed=0, device="cpu", log=print):
    """Train on the windows in train_idx; keep the epoch with the best select_score on the validation recordings."""
    torch.manual_seed(seed); np.random.seed(seed)
    W = torch.as_tensor(waves[train_idx], device=device).unsqueeze(1)
    Fe = torch.as_tensor(feats[train_idx], device=device)
    Y = torch.as_tensor(y_win[train_idx], device=device)
    counts = np.bincount(y_win[train_idx], minlength=4).astype(float)
    weight = torch.tensor(1 / np.sqrt(np.maximum(counts, 1)), dtype=torch.float32, device=device)
    model = AFNet(feats.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    steps = epochs * int(np.ceil(len(Y) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss(weight=weight / weight.sum() * 4)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(Y), device=device)
        for i in range(0, len(Y), batch):
            idx = perm[i:i + batch]
            w = W[idx]
            sign = torch.where(torch.rand(len(idx), 1, 1, device=device) < 0.5, -1.0, 1.0)     # polarity
            w = w * sign * (0.8 + 0.4 * torch.rand(len(idx), 1, 1, device=device))           # amplitude
            w = w + 0.03 * torch.randn_like(w)                                                  # noise
            loss = loss_fn(model(w, Fe[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        pv = predict_windows(model, waves[val_idx], feats[val_idx], device)
        P = per_recording(pv, owner[val_idx], len(y_rec))[val_recs]
        score = select_score(y_rec[val_recs], np.array(CLASSES)[P.argmax(1)])
        log(f"epoch {ep:2d} | loss {loss.item():.3f} | val mean F1 (4 classes) {score:.3f} | {time.time() - t0:.0f} s")
        if score > best:
            best, best_ep = score, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu().eval(), best, best_ep


def save_members(path, members, scaler, meta):
    torch.save({"members": [m.state_dict() for m in members], "n_feat": int(scaler.mu.shape[0]),
                "mu": scaler.mu.tolist(), "sd": scaler.sd.tolist(), **meta}, path)


def load_members(path):
    ck = torch.load(path, map_location="cpu")
    members = []
    for sd in ck["members"]:
        m = AFNet(ck["n_feat"])
        m.load_state_dict(sd)
        members.append(m.eval())
    scaler = FeatureScaler()
    scaler.mu, scaler.sd = np.asarray(ck["mu"]), np.asarray(ck["sd"])
    return members, scaler, ck
