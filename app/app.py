"""ECG AF Detector — Gradio app (CPU). Classifies every 30-s window of a single-lead ECG as Normal rhythm,
atrial fibrillation (AF), Other abnormal rhythm or Too noisy, and summarizes the AF burden of the recording."""
import inspect
import json
import os
import sys
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "src")):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
from afsuite.data import CLASSES
from afsuite.detect import bandpass, detect_r_peaks
from afsuite.cnn import load_members, make_windows, predict_windows, window_starts

COLORS = {"N": "#2e9e5b", "A": "#dc2626", "O": "#f39c12", "~": "#9ca3af"}
SHORT = {"N": "Normal", "A": "AF", "O": "Other", "~": "Noisy"}
LONG = {"N": "Normal rhythm", "A": "Atrial fibrillation", "O": "Other abnormal rhythm", "~": "Too noisy to read"}
VIEW, MAX_HOURS = 10, 4
GRID_MAJOR, GRID_MINOR, PAPER = "rgba(220,60,60,0.55)", "rgba(220,60,60,0.18)", "#fff6f6"
FONT = dict(family="Inter, sans-serif", size=12)

ICON = {
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "bars": '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
    "repeat": '<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/><polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "signal": '<path d="M2 12h3l3-8 4 16 3-8h7"/>',
    "usercheck": '<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><polyline points="17 11 19 13 23 9"/>',
}

CSS = """
.title {text-align:center; font-size:2.1rem; font-weight:800; margin:8px 0 12px; letter-spacing:-0.02em;}
.sec {font-weight:700; font-size:1.05rem; margin:20px 0 8px; padding-bottom:6px;
      border-bottom:2px solid var(--border-color-primary);}
.kpi-row {display:flex; gap:12px; flex-wrap:wrap; margin:6px 0 12px;}
.kpi {flex:1 1 170px; display:flex; gap:12px; align-items:center; border:1px solid var(--border-color-primary);
      border-radius:14px; padding:12px 14px; background:var(--background-fill-secondary);}
.kpi .ic {flex:0 0 42px; height:42px; border-radius:12px; display:flex; align-items:center; justify-content:center;
          background:rgba(225,29,72,.08); color:#e11d48;}
.kpi.ok .ic {background:rgba(46,158,91,.12); color:#2e9e5b;}
.kpi.warn .ic {background:rgba(243,156,18,.15); color:#d68910;}
.kpi.bad .ic {background:rgba(220,38,38,.12); color:#dc2626;}
.kpi .ic svg {width:22px; height:22px; stroke:currentColor; fill:none; stroke-width:2;
              stroke-linecap:round; stroke-linejoin:round;}
.kpi .v {font-size:1.4rem; font-weight:700; line-height:1.15;}
.kpi .l {font-size:.8rem; opacity:.8;}
.kpi .s {font-size:.72rem; opacity:.6;}
.req {display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 12px;}
.req span {padding:4px 10px; border-radius:999px; font-size:.82rem;
           border:1px solid var(--border-color-primary); background:var(--background-fill-secondary);}
.step {font-weight:700; font-size:1rem; margin:4px 0 2px;}
.pipe {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:8px 0 16px;}
.chip {padding:6px 12px; border-radius:999px; border:1px solid var(--border-color-primary);
       background:var(--background-fill-secondary); font-size:.9rem;}
.arrow {opacity:.5;}
table td, table th {font-family: var(--font) !important;}
"""
DESC = ("Classifies the heart rhythm in every 30 seconds of a single-lead ECG as normal, atrial fibrillation (AF), "
        "another abnormal rhythm, or too noisy to read, and reports how much of the recording was in AF.")
REQ_HTML = ('<div class="req"><span>Single-lead ECG</span><span>Any lead (trained on lead I-like, tested on lead II-like)</span>'
            '<span>CSV · millivolts or raw ADC values</span><span>Any sampling rate ≥ 100 Hz</span>'
            '<span>9 s to 4 h</span><span>Research tool · not a medical device</span></div>')
PIPE_HTML = ('<div class="pipe">' + '<span class="arrow">→</span>'.join(
    f'<span class="chip">{s}</span>' for s in
    ["Raw ECG", "Auto-scale & polarity check", "R-peak detection", "30-s windows (15-s step)",
     "35 rhythm / beat-shape / quality features", "5 × hybrid 1D-CNN", "Rhythm per window + AF burden"]) + "</div>")
INPUT_MD = """
### Which recordings work?
| Source | Works? |
|---|---|
| Handheld single-lead devices (AliveCor-like, fingers, lead I) | ✅ The training data came from this kind of device |
| Holter recordings, chest electrodes, lead II-like (MIT-BIH) | ✅ Tested on 45 external patients (see results) |
| AD8232 or another 3-electrode board | ⚠️ Expected to work with a lead II-like placement; accuracy not measured |
| 12-lead hospital ECG | ✅ Choose one column (lead II recommended) |

Upload a CSV with one column per lead (a header row is optional) and enter its sampling rate.
"""


def sec(title):
    return f'<div class="sec">{title}</div>'


def hms(s):
    s = max(float(s), 0.0)
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}" if s >= 3600 else f"{int(s // 60):02d}:{s % 60:04.1f}"


def nice_dur(s):
    return f"{s / 3600:.1f} h" if s >= 3600 else (f"{s / 60:.1f} min" if s >= 120 else f"{s:.0f} s")


def cards_html(items):
    out = []
    for icon, label, value, sub, status in items:
        out.append(f'<div class="kpi {status}"><div class="ic"><svg viewBox="0 0 24 24">{ICON[icon]}</svg></div>'
                   f'<div><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div></div>')
    return '<div class="kpi-row">' + "".join(out) + "</div>"


def read_columns(path):
    head = pd.read_csv(path, nrows=5, header=None)
    if pd.to_numeric(head.iloc[0], errors="coerce").isna().all():
        return [str(v) for v in head.iloc[0]], True
    return [f"Column {i + 1}" for i in range(head.shape[1])], False


def default_column(names):
    for n in names:
        if n.strip().upper().replace("_", " ") in ("II", "MLII", "LEAD II", "ECG II", "ECG"):
            return n
    return names[-1]


def read_signal(path, column):
    names, has_header = read_columns(path)
    df = pd.read_csv(path, header=0 if has_header else None)
    idx = names.index(column) if column in names else len(names) - 1
    return pd.to_numeric(df.iloc[:, idx], errors="coerce").dropna().to_numpy(float)


def auto_adjust(sig, fs):
    """Rescale signals that are not in millivolts (for example raw ADC counts) and invert upside-down ones."""
    notes = []
    f = bandpass(sig, fs)
    hi, lo = np.percentile(f, 99.5), np.percentile(f, 0.5)
    if -lo > 1.5 * hi:
        sig, f = -sig, -f
        notes.append("signal inverted (R waves pointed down)")
    amp = np.percentile(np.abs(f), 99.5)
    if amp > 5 or amp < 0.3:
        k = 1.2 / amp
        sig = sig * k
        notes.append(f"amplitude rescaled ×{k:.3g} (values did not look like millivolts)")
    return sig, notes


# ---------------------------------------------------------------- examples
def load_examples(path):
    if not os.path.exists(path):
        return [], {}
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return [(m["label"], m["key"]) for m in meta], {m["key"]: dict(m, signal=z[m["key"]] / 1000.0) for m in meta}


def af_fraction(starts, intervals, win=30.0):
    return np.array([sum(max(0.0, min(t + win, e) - max(t, s)) for s, e in intervals) / win for t in starts])


# ---------------------------------------------------------------- analysis
def analyze(sig, fs, members, scaler):
    peaks = detect_r_peaks(sig, fs)
    waves, feats = make_windows(sig, fs, peaks)
    F = scaler(pd.DataFrame(feats).to_numpy(float))
    with torch.no_grad():
        P = np.mean([predict_windows(m, waves, F, "cpu") for m in members], 0)
    dur = len(sig) / fs
    starts = np.array(window_starts(len(sig), fs))
    ends = np.r_[starts[1:], dur]                              # each window owns the time up to the next start
    cls = np.array(CLASSES)[P.argmax(1)]
    return dict(signal=bandpass(sig, fs), fs=fs, peaks=peaks, starts=starts, ends=ends, probs=P, cls=cls, dur=dur)


def runs(res, code):
    """Consecutive windows of one class -> list of (start s, end s)."""
    out, i, cls = [], 0, res["cls"]
    while i < len(cls):
        if cls[i] == code:
            j = i
            while j + 1 < len(cls) and cls[j + 1] == code:
                j += 1
            out.append((res["starts"][i], res["ends"][j]))
            i = j + 1
        else:
            i += 1
    return out


def summarize(res):
    span = res["ends"] - res["starts"]
    share = {c: 100 * span[res["cls"] == c].sum() / max(span.sum(), 1e-9) for c in CLASSES}
    rr = np.diff(res["peaks"]) / res["fs"]
    t_rr = res["peaks"][1:] / res["fs"]
    ok = (rr > 0.25) & (rr < 2.5)
    rr, t_rr = rr[ok], t_rr[ok]
    sm = pd.Series(60 / rr).rolling(10, center=True, min_periods=5).median().to_numpy() if len(rr) else np.array([])
    ep = runs(res, "A")
    longest = max((e - s for s, e in ep), default=0.0)
    single = len(res["cls"]) == 1
    conf = res["probs"].max(1)
    return dict(share=share, rr=rr, t_rr=t_rr, hr=60 / rr.mean() if len(rr) else float("nan"),
                hr_min=np.nanmin(sm) if len(sm) else float("nan"), hr_max=np.nanmax(sm) if len(sm) else float("nan"),
                irregular=100 * np.mean(np.abs(np.diff(rr)) / rr[:-1] > 0.15) if len(rr) > 2 else float("nan"),
                episodes=ep, longest=longest, single=single, conf0=conf[0], n_win=len(res["cls"]))


def window_class_at(res, t):
    i = np.searchsorted(res["starts"], t, side="right") - 1
    return res["cls"][np.clip(i, 0, len(res["cls"]) - 1)], np.clip(i, 0, len(res["cls"]) - 1)


# ---------------------------------------------------------------- figures
def unit_of(res):
    return (60.0, "time (min)") if res["dur"] > 300 else (1.0, "time (s)")


def timeline_fig(res, t0=None, truth=None, height=150):
    u, xlab = unit_of(res)
    fig = go.Figure()
    rows = [("Model", 0.0)] + ([("Cardiologists", 1.25)] if truth is not None else [])
    for c in CLASSES:
        for s, e in runs(res, c):
            fig.add_shape(type="rect", x0=s / u, x1=e / u, y0=0, y1=1, fillcolor=COLORS[c], line_width=0, layer="below")
    if truth is not None:
        fig.add_shape(type="rect", x0=0, x1=res["dur"] / u, y0=1.25, y1=2.25, fillcolor="#e5e7eb", line_width=0)
        for s, e in truth:
            fig.add_shape(type="rect", x0=s / u, x1=e / u, y0=1.25, y1=2.25, fillcolor=COLORS["A"], line_width=0)
    for c in CLASSES:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=COLORS[c], symbol="square"),
                                 name=LONG[c]))
    if t0 is not None:
        fig.add_shape(type="rect", x0=t0 / u, x1=(t0 + VIEW) / u, y0=-0.12, y1=(2.37 if truth is not None else 1.12),
                      line=dict(color="#111", width=2), fillcolor="rgba(0,0,0,0)")
    top = 2.25 if truth is not None else 1
    fig.update_yaxes(range=[-0.2, top + 0.2], tickvals=[0.5] + ([1.75] if truth is not None else []),
                     ticktext=[r[0] for r in rows], showgrid=False, zeroline=False, fixedrange=True)
    fig.update_xaxes(range=[0, res["dur"] / u], title_text=xlab, showgrid=False, zeroline=False)
    fig.update_layout(template="plotly_white", height=height + (40 if truth is not None else 0),
                      margin=dict(l=95, r=20, t=30, b=40), font=FONT, hovermode=False,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def ecg_paper(fig, a_s, b_s, ylim):
    xt = np.arange(np.floor(a_s / 0.2) * 0.2, b_s + 1e-9, 0.2)
    yt = np.arange(-ylim, ylim + 1e-9, 0.5)
    lab = lambda v: f"{v:.0f}" if abs(v - round(v)) < 1e-6 else ""
    fig.update_xaxes(range=[a_s, b_s], tickmode="array", tickvals=xt, ticktext=[lab(v) for v in xt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.04, showgrid=True, gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="time (s)")
    fig.update_yaxes(range=[-ylim, ylim], tickmode="array", tickvals=yt, ticktext=[lab(v) for v in yt],
                     showgrid=True, gridcolor=GRID_MAJOR, gridwidth=1, zeroline=False, ticks="",
                     minor=dict(tickmode="linear", tick0=0, dtick=0.1, showgrid=True, gridcolor=GRID_MINOR, gridwidth=0.5),
                     title_text="mV")
    fig.update_layout(plot_bgcolor=PAPER, paper_bgcolor="white", height=340, margin=dict(l=55, r=20, t=50, b=45),
                      font=FONT, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"))
    return fig


def ecg_fig(res, t0, markers=False):
    x, fs = res["signal"], res["fs"]
    a, b = int(t0 * fs), min(int((t0 + VIEW) * fs), len(x))
    ylim = float(np.clip(np.ceil(np.percentile(np.abs(x), 99.9) / 0.5) * 0.5 + 0.5, 1.5, 4.0))
    fig = go.Figure(go.Scatter(x=np.arange(a, b) / fs, y=x[a:b], mode="lines", line=dict(color="#111", width=1.3),
                               hoverinfo="skip", showlegend=False))
    if markers:
        p = res["peaks"][(res["peaks"] >= a) & (res["peaks"] < b)]
        if len(p):
            idx = [window_class_at(res, q / fs)[1] for q in p]
            cls = res["cls"][idx]
            pr = 100 * res["probs"][idx]
            for c in CLASSES:
                sel = cls == c
                if sel.any():
                    fig.add_trace(go.Scatter(
                        x=p[sel] / fs, y=x[p[sel]], mode="markers", name=f"beat in a window classified {SHORT[c]}",
                        marker=dict(size=10, color=COLORS[c], line=dict(width=1.5, color="white")),
                        customdata=pr[sel],
                        hovertemplate=("window: Normal %{customdata[0]:.0f}% · AF %{customdata[1]:.0f}% · "
                                       "Other %{customdata[2]:.0f}% · Noisy %{customdata[3]:.0f}%<extra></extra>")))
    return ecg_paper(fig, t0, t0 + VIEW, ylim)


def prob_fig(res, truth=None):
    u, xlab = unit_of(res)
    t = (res["starts"] + np.minimum(res["starts"] + 30, res["dur"])) / 2 / u
    fig = go.Figure()
    if truth is not None:
        for s, e in truth:
            fig.add_vrect(x0=s / u, x1=e / u, fillcolor=COLORS["A"], opacity=0.08, line_width=0)
    for k, c in enumerate(CLASSES):
        fig.add_trace(go.Scatter(x=t, y=100 * res["probs"][:, k], mode="lines", stackgroup="p", name=LONG[c],
                                 line=dict(width=0.5, color=COLORS[c])))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=55, r=10, t=40, b=45), font=FONT,
                      title=dict(text="Model probability of each rhythm, window by window"
                                 + (" (shaded: AF per cardiologists)" if truth is not None else ""), font=dict(size=14)),
                      xaxis_title=xlab, yaxis=dict(title="%", range=[0, 100]),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
    return fig


def tacho_fig(res, s):
    u, xlab = unit_of(res)
    cls = np.array([window_class_at(res, t)[0] for t in s["t_rr"]]) if len(s["rr"]) else np.array([])
    fig = go.Figure()
    for c in CLASSES:
        sel = cls == c
        if sel.any():
            fig.add_trace(go.Scattergl(x=s["t_rr"][sel] / u, y=1000 * s["rr"][sel], mode="markers", name=SHORT[c],
                                       marker=dict(size=4, color=COLORS[c])))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=55, r=10, t=40, b=45), font=FONT,
                      title=dict(text="Beat-to-beat interval (RR) across the recording", font=dict(size=14)),
                      xaxis_title=xlab, yaxis_title="RR interval (ms)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
    return fig


def poincare_fig(res, s):
    rr = 1000 * s["rr"]
    fig = go.Figure()
    if len(rr) > 2:
        cls = np.array([window_class_at(res, t)[0] for t in s["t_rr"][:-1]])
        for c in CLASSES:
            sel = cls == c
            if sel.any():
                fig.add_trace(go.Scattergl(x=rr[:-1][sel], y=rr[1:][sel], mode="markers", name=SHORT[c],
                                           marker=dict(size=4, color=COLORS[c], opacity=0.6)))
        lim = [max(200, np.percentile(rr, 0.5) - 50), min(2500, np.percentile(rr, 99.5) + 50)]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="#999", dash="dash", width=1),
                                 showlegend=False, hoverinfo="skip"))
        fig.update_xaxes(range=lim); fig.update_yaxes(range=lim, scaleanchor="x")
    fig.update_layout(template="plotly_white", height=360, margin=dict(l=55, r=10, t=40, b=45), font=FONT,
                      title=dict(text="Poincaré plot: each RR interval against the next", font=dict(size=14)),
                      xaxis_title="RR n (ms)", yaxis_title="RR n+1 (ms)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"))
    return fig


# ---------------------------------------------------------------- cards and text
def status_hr(hr):
    return "ok" if 60 <= hr <= 100 else "warn"


def quality(s):
    noisy = s["share"]["~"]
    return ("Good", "ok") if noisy < 5 else (("Fair", "warn") if noisy < 20 else ("Poor", "bad"))


def basic_cards(res, s):
    q, qs = quality(s)
    items = [("clock", "Recording length", nice_dur(res["dur"]), "", ""),
             ("heart", "Average heart rate", f"{s['hr']:.0f} bpm", "usual at rest: 60–100", status_hr(s["hr"]))]
    if s["single"]:
        c = res["cls"][0]
        items.append(("pulse", "Rhythm", LONG[c], f"model confidence {100 * s['conf0']:.0f}%",
                      {"N": "ok", "A": "bad", "O": "warn", "~": ""}[c]))
    else:
        af = s["share"]["A"]
        items.append(("pulse", "Atrial fibrillation", f"{af:.0f}% of the time" if af > 0 else "Not detected",
                      "share of the recording", "bad" if af > 0 else "ok"))
        items.append(("zap", "AF episodes", f"{len(s['episodes'])}",
                      f"longest {nice_dur(s['longest'])}" if s["episodes"] else "none", "bad" if s["episodes"] else "ok"))
    items.append(("signal", "Signal quality", q, f"{s['share']['~']:.0f}% too noisy to read", qs))
    return cards_html(items)


def rhythm_cards(s):
    sh = s["share"]
    return cards_html([("check", "Normal rhythm", f"{sh['N']:.1f}%", "of the time", "ok"),
                       ("pulse", "Atrial fibrillation", f"{sh['A']:.1f}%", "of the time", "bad" if sh["A"] > 0 else "ok"),
                       ("alert", "Other abnormal rhythm", f"{sh['O']:.1f}%", "of the time", "warn" if sh["O"] > 0 else "ok"),
                       ("signal", "Too noisy", f"{sh['~']:.1f}%", "of the time", "")])


def hr_cards(s):
    return cards_html([("heart", "Average heart rate", f"{s['hr']:.0f} bpm", "usual at rest: 60–100", status_hr(s["hr"])),
                       ("bars", "Lowest / highest rate", f"{s['hr_min']:.0f} / {s['hr_max']:.0f} bpm", "10-beat median", ""),
                       ("repeat", "Irregular beat-to-beat intervals", f"{s['irregular']:.0f}%",
                        "of intervals change by more than 15%", "warn" if s["irregular"] > 20 else "ok")])


def summary_md(title, res, s, notes):
    lines = [f"**{title}**",
             f"**{nice_dur(res['dur'])}** analyzed in **{s['n_win']} window(s)** of 30 s (15-s step); "
             f"average heart rate **{s['hr']:.0f} bpm**."]
    if s["single"]:
        c = res["cls"][0]
        lines.append(f"The model reads this recording as **{LONG[c].lower()}** (confidence {100 * s['conf0']:.0f}%).")
    elif s["episodes"]:
        lines.append(f"**Atrial fibrillation in {s['share']['A']:.1f}% of the recording**: {len(s['episodes'])} episode(s), "
                     f"the longest about {nice_dur(s['longest'])}.")
    else:
        lines.append("**No atrial fibrillation detected.**")
    if s["share"]["O"] > 0:
        lines.append(f"Other abnormal rhythm in {s['share']['O']:.1f}% of the time.")
    if s["share"]["~"] > 0:
        lines.append(f"{s['share']['~']:.1f}% of the time was too noisy to read.")
    if notes:
        lines.append("*Automatic adjustment: " + "; ".join(notes) + ".*")
    return "\n\n".join(lines)


def validation(res, truth, true_label=None):
    if true_label is not None:
        c = res["cls"][0] if len(res["cls"]) == 1 else max(CLASSES, key=lambda k: (res["cls"] == k).sum())
        ok = c == true_label
        return (cards_html([("usercheck", "Expert label of this recording", LONG[true_label], "PhysioNet/CinC 2017", ""),
                            ("check" if ok else "alert", "Model", LONG[c], "agrees" if ok else "disagrees", "ok" if ok else "bad")]),
                "")
    if truth is None:
        return "", "*Agreement with cardiologists is shown for the example recordings only.*"
    af = af_fraction(res["starts"], truth) > 0.5
    pred = res["cls"] == "A"
    span = res["ends"] - res["starts"]
    true_burden = 100 * span[af].sum() / span.sum()
    agree = 100 * np.mean(af == pred)
    return (cards_html([("usercheck", "Agreement with cardiologists", f"{agree:.1f}%", "AF / not AF, window by window",
                         "ok" if agree >= 90 else "warn"),
                        ("pulse", "AF burden: cardiologists", f"{true_burden:.1f}%", "of the time", ""),
                        ("pulse", "AF burden: model", f"{100 * span[pred].sum() / span.sum():.1f}%", "of the time", "")]),
            "")


def table_and_csv(res):
    P = res["probs"]
    df = pd.DataFrame({"start": [hms(t) for t in res["starts"]], "end": [hms(t) for t in np.minimum(res["starts"] + 30, res["dur"])],
                       "rhythm": [LONG[c] for c in res["cls"]],
                       "Normal %": (100 * P[:, 0]).round(0), "AF %": (100 * P[:, 1]).round(0),
                       "Other %": (100 * P[:, 2]).round(0), "Noisy %": (100 * P[:, 3]).round(0)})
    f = tempfile.NamedTemporaryFile(delete=False, suffix="_rhythm_windows.csv")
    out = df.copy()
    out.insert(0, "start_s", res["starts"].round(1))
    out.to_csv(f.name, index=False)
    return df.head(500), f.name


# ---------------------------------------------------------------- app
def build_demo(model_path, examples_path, about_md):
    members, scaler, _ = load_members(model_path)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    choices, EX = load_examples(examples_path)

    def run(source, ex_key, upload, column, fs):
        notes, truth, true_label = [], None, None
        if source == "Upload my ECG":
            if not upload:
                raise gr.Error("Upload a CSV file first, or switch to 'Example'.")
            fs = float(fs or 0)
            if fs < 100:
                raise gr.Error("Please enter the sampling rate of your file (at least 100 Hz).")
            sig = read_signal(upload, column)
            if len(sig) < 9 * fs:
                raise gr.Error("The recording is shorter than 9 seconds.")
            if len(sig) > MAX_HOURS * 3600 * fs:
                gr.Warning(f"Longer than {MAX_HOURS} h: only the first {MAX_HOURS} h are analyzed.")
                sig = sig[: int(MAX_HOURS * 3600 * fs)]
            sig, notes = auto_adjust(sig, fs)
            title = f"Your upload · {nice_dur(len(sig) / fs)} at {fs:g} Hz · column '{column}'"
        else:
            ex = EX[ex_key]
            sig, fs, title = ex["signal"], float(ex["fs"]), ex["label"]
            truth = [tuple(v) for v in ex.get("af", [])] if ex.get("truth") == "intervals" else None
            true_label = ex.get("rec_label")
        res = analyze(sig, fs, members, scaler)
        res.update(title=title, notes=notes, truth=truth, true_label=true_label, t0=first_view(res))
        return res

    def first_view(res):
        ep = runs(res, "A")
        return float(max(0.0, min(ep[0][0], res["dur"] - VIEW))) if ep else 0.0

    def position(res):
        return f"<center>Showing <b>{hms(res['t0'])} – {hms(min(res['t0'] + VIEW, res['dur']))}</b> of {hms(res['dur'])}</center>"

    def views(res):
        return (timeline_fig(res, res["t0"]), ecg_fig(res, res["t0"]), position(res),
                timeline_fig(res, res["t0"], res["truth"]), ecg_fig(res, res["t0"], markers=True), position(res), res)

    def shift(res, step):
        if res is None:
            raise gr.Error("Analyze an ECG first.")
        res["t0"] = float(np.clip(res["t0"] + step, 0, max(res["dur"] - VIEW, 0)))
        return views(res)

    def next_af(res):
        if res is None:
            raise gr.Error("Analyze an ECG first.")
        ep = [s for s, e in runs(res, "A") if s > res["t0"] + 1] or [s for s, e in runs(res, "A")]
        if not ep:
            gr.Info("No atrial fibrillation in this recording.")
            return views(res)
        res["t0"] = float(np.clip(ep[0], 0, max(res["dur"] - VIEW, 0)))
        return views(res)

    def render(res):
        s = summarize(res)
        val_cards, val_md = validation(res, res["truth"], res["true_label"])
        tbl, csv = table_and_csv(res)
        return (basic_cards(res, s), timeline_fig(res, res["t0"]), ecg_fig(res, res["t0"]), position(res),
                summary_md(res["title"], res, s, res["notes"]), timeline_fig(res, res["t0"], res["truth"]),
                ecg_fig(res, res["t0"], markers=True), position(res), rhythm_cards(s), prob_fig(res, res["truth"]),
                hr_cards(s), tacho_fig(res, s), poincare_fig(res, s), val_cards, val_md, tbl, csv, res)

    def on_upload(f):
        if not f:
            return gr.update(choices=[], value=None)
        names, _ = read_columns(f)
        return gr.update(choices=names, value=default_column(names))

    theme = gr.themes.Soft(primary_hue="rose", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"])
    if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
        blocks_kw, launch_kw = dict(theme=theme, css=CSS), {}
    else:
        blocks_kw, launch_kw = {}, dict(theme=theme, css=CSS)
    default = choices[0][1] if choices else None

    with gr.Blocks(title="ECG AF Detector", **blocks_kw) as demo:
        state = gr.State()
        gr.HTML('<div class="title">ECG AF Detector</div>')
        with gr.Group():
            with gr.Row(equal_height=False):
                with gr.Column(scale=2):
                    gr.HTML('<div class="step">1 · Choose an ECG</div>')
                    source = gr.Radio(["Example", "Upload my ECG"], value="Example", show_label=False)
                    example = gr.Dropdown(choices, value=default, label="Example recording (none used for training)")
                    with gr.Row(visible=False) as up_row:
                        upload = gr.File(label="Your ECG file (.csv)", file_types=[".csv", ".txt"], type="filepath", scale=2)
                        with gr.Column(scale=1):
                            column = gr.Dropdown([], label="ECG column", allow_custom_value=True)
                            fs = gr.Number(300, label="Sampling rate (Hz)", precision=0)
                with gr.Column(scale=1):
                    gr.HTML('<div class="step">2 · Analyze</div>')
                    run_btn = gr.Button("Analyze ECG", variant="primary", size="lg")
        with gr.Tabs():
            with gr.Tab("Basic"):
                b_cards = gr.HTML()
                b_time = gr.Plot(show_label=False)
                b_ecg = gr.Plot(show_label=False)
                with gr.Row():
                    b_prev = gr.Button("◀ 10 s", size="sm")
                    b_pos = gr.Markdown()
                    b_next = gr.Button("10 s ▶", size="sm")
            with gr.Tab("Advanced"):
                gr.HTML(sec("Recording"))
                a_summary = gr.Markdown()
                gr.HTML(sec("Rhythm over time"))
                a_time = gr.Plot(show_label=False)
                a_rhythm = gr.HTML()
                a_prob = gr.Plot(show_label=False)
                gr.HTML(sec("Signal viewer"))
                a_ecg = gr.Plot(show_label=False)
                with gr.Row():
                    a_prev = gr.Button("◀ 10 s", size="sm")
                    a_pos = gr.Markdown()
                    a_next = gr.Button("10 s ▶", size="sm")
                    a_af = gr.Button("Next AF episode ⏭", size="sm", variant="secondary")
                gr.HTML(sec("Heart rate & beat-to-beat intervals"))
                a_hr = gr.HTML()
                with gr.Row():
                    a_tacho = gr.Plot(show_label=False)
                    a_poinc = gr.Plot(show_label=False)
                gr.HTML(sec("Comparison with the experts"))
                a_val = gr.HTML()
                a_val_md = gr.Markdown()
                gr.HTML(sec("Window by window"))
                a_tbl = gr.Dataframe(interactive=False, label="First 500 windows")
                a_csv = gr.File(label="All windows (CSV)")
            with gr.Tab("How it works"):
                gr.Markdown(DESC)
                gr.HTML(REQ_HTML)
                gr.HTML(PIPE_HTML)
                gr.Markdown(INPUT_MD)
                gr.Markdown(about_md)
        gr.Markdown("<center><small>Fady Mostafa Omar · 2026 · Data: PhysioNet/CinC Challenge 2017, MIT-BIH Arrhythmia "
                    "Database, MIT-BIH Atrial Fibrillation Database (PhysioNet)</small></center>")

        outs = [b_cards, b_time, b_ecg, b_pos, a_summary, a_time, a_ecg, a_pos, a_rhythm, a_prob, a_hr, a_tacho, a_poinc,
                a_val, a_val_md, a_tbl, a_csv, state]
        nav = [b_time, b_ecg, b_pos, a_time, a_ecg, a_pos, state]
        source.change(lambda s: (gr.update(visible=s == "Example"), gr.update(visible=s != "Example")),
                      source, [example, up_row])
        upload.change(on_upload, upload, column)
        run_btn.click(lambda s, e, u, c, f: render(run(s, e, u, c, f)), [source, example, upload, column, fs], outs)
        example.change(lambda e: render(run("Example", e, None, None, None)), example, outs)
        if default is not None:
            demo.load(lambda: render(run("Example", default, None, None, None)), None, outs)
        for btn, fn in ((b_prev, lambda r: shift(r, -VIEW)), (b_next, lambda r: shift(r, VIEW)),
                        (a_prev, lambda r: shift(r, -VIEW)), (a_next, lambda r: shift(r, VIEW)), (a_af, next_af)):
            btn.click(fn, state, nav)
    return demo, launch_kw


def build_about(results_dir):
    """'How it works' text, filled from the result files of Steps 2.3-2.5."""
    t = pd.read_csv(os.path.join(results_dir, "cnn_vs_baseline_test.csv"), index_col=0)
    bs = pd.read_csv(os.path.join(results_dir, "bootstrap_cnn_vs_gb.csv")).iloc[0]
    ext = pd.read_csv(os.path.join(results_dir, "external_windows.csv"))
    bur = pd.read_csv(os.path.join(results_dir, "external_burden.csv"))
    rows = "\n".join(f"| {i} | " + " | ".join(str(v) for v in t.loc[i]) + " |" for i in t.index)
    e = ext[ext["model"] == "CNN ensemble"]
    erows = "\n".join(f"| {r['database']} | {int(r['windows']):,} | {r['Se %']:.1f} | {r['PPV %']:.1f} | "
                      f"{r['Specificity %']:.1f} |" for _, r in e.iterrows())
    brows = []
    for db, name in (("afdb", "MIT-BIH AF DB"), ("mitdb", "MIT-BIH DS2")):
        b = bur[bur["database"] == db]
        brows.append(f"| {name} | {len(b)} | {np.mean(np.abs(b['CNN AF %'] - b['true AF %'])):.1f} | "
                     f"{np.corrcoef(b['true AF %'], b['CNN AF %'])[0, 1]:.2f} |")
    fa = bur[(bur["database"] == "mitdb") & (bur["true AF %"] < 1) & (bur["CNN AF %"] > 10)]["record"].astype(str).tolist()
    return f"""
### The four rhythm classes
| Class | Meaning |
|---|---|
| **Normal** | Normal sinus rhythm |
| **AF** | Atrial fibrillation: chaotic atrial activity, irregularly irregular beat timing, no P waves |
| **Other** | Any other abnormal rhythm (for example frequent extra beats or other arrhythmias) |
| **Noisy** | Too noisy to classify |

### How it was tested
Trained on the PhysioNet/CinC Challenge 2017 recordings (single lead, 300 Hz, AliveCor). A stratified 20% of the
recordings was locked away as a test set before any model was built. Official score = mean F1 of Normal, AF and Other.

| Locked test set (1,706 recordings) | {" | ".join(t.columns)} |
|---|{"---|" * len(t.columns)}
{rows}

The ensemble minus gradient-boosting difference is {bs['diff']:+.3f} (95% bootstrap interval {bs['ci_low']:+.3f} to
{bs['ci_high']:+.3f}), so on this test set the two are **not clearly different**. The 2017 winners reported 0.83 on the
Challenge's hidden test set, which is not public, so the numbers are not directly comparable.

### External test: recordings from other devices and patients, never used for training
Every 30-s window (15-s step) was compared with the cardiologists' rhythm annotations (AF = more than half of the window).

| Database | Windows | AF sensitivity % | AF positive predictivity % | Specificity % |
|---|---|---|---|---|
{erows}

| AF burden per patient | Patients | Mean absolute error (points) | Correlation r |
|---|---|---|---|
{chr(10).join(brows)}

### Limitations
- **Frequent extra beats can look like AF.** In MIT-BIH records {", ".join(fa) or "none"} (no AF annotated), the model
  reported AF for part of the time; irregular timing caused by extra beats mimics AF.
- Atrial flutter counts as "not AF" in the external tests; the model may call it AF or Other.
- CinC 2017 has no patient identifiers, so the internal split cannot guarantee that no person appears in both parts;
  the labels were revised by the organizers several times, so some are uncertain.
- AD8232 recordings are expected to work with a lead II-like placement, but accuracy on them has not been measured.
- Research and education tool, not a medical device.
"""


def main():
    import argparse
    root = os.path.join(HERE, "..")
    p = argparse.ArgumentParser(description="ECG AF Detector (Gradio app)")
    p.add_argument("--model", default=os.path.join(root, "models", "afnet_ensemble.pt"))
    p.add_argument("--examples", default=os.path.join(HERE, "examples", "examples.npz"))
    p.add_argument("--results", default=os.path.join(root, "results"))
    p.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    a = p.parse_args()
    demo, launch_kw = build_demo(a.model, a.examples, build_about(a.results))
    demo.launch(share=a.share, **launch_kw)


if __name__ == "__main__":
    main()
