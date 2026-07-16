"""Conditional probabilities of the open: {prev day up/down} x {gap up/down}.

Outcomes per state (SPY+QQQ, 5m bars 2016-2026):
  P(open->close up), avg open->close, P(close green vs prev close),
  P(recovers to prev close intraday at any point),
  time-of-day of session HIGH and LOW: P(first 30m), P(first hour), P(last hour), median.
Also gap-magnitude buckets for the down-down state (the user's lead case).
"""
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")


def load(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts")
    df["day"] = df["ts"].dt.date
    df["mins"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute) - 570  # mins since open
    return df


def daystats(df):
    rows = []
    for dy, g in df.groupby("day"):
        if len(g) < 60:
            continue
        o = g["open"].iloc[0]; c = g["close"].iloc[-1]
        hi_i = g["high"].idxmax(); lo_i = g["low"].idxmin()
        rows.append(dict(day=dy, o=o, c=c, h=g["high"].max(), l=g["low"].min(),
                         t_hi=g.loc[hi_i, "mins"], t_lo=g.loc[lo_i, "mins"]))
    d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    d["pc"] = d["c"].shift(1)              # prev close
    d["prev_dn"] = d["c"].shift(1) < d["c"].shift(2)
    d["gap"] = d["o"] / d["pc"] - 1
    d["oc"] = d["c"] / d["o"] - 1
    d["green"] = d["c"] > d["pc"]
    d["touch_pc"] = d["h"] >= d["pc"]      # recovered to prev close at some point (for gap-down)
    return d.dropna(subset=["pc", "gap"])


def timeprofile(S, col):
    a = S[col].values
    return (f"P(1st30m) {np.mean(a<=30):.0%}  P(1st hr) {np.mean(a<=60):.0%}  "
            f"P(last hr) {np.mean(a>=330):.0%}  med {np.median(a):.0f}m")


for sym in ["SPY", "QQQ"]:
    d = daystats(load(sym))
    print(f"\n================ {sym} (n={len(d)} days) ================")
    print(f"{'STATE':<28s}{'n':>5s}{'P(O->C up)':>11s}{'avg O->C':>10s}{'P(green)':>10s}")
    states = [
        ("prev DOWN + gap DOWN", (d.prev_dn) & (d.gap < 0)),
        ("prev DOWN + gap UP", (d.prev_dn) & (d.gap >= 0)),
        ("prev UP   + gap DOWN", (~d.prev_dn) & (d.gap < 0)),
        ("prev UP   + gap UP", (~d.prev_dn) & (d.gap >= 0)),
    ]
    for lbl, m in states:
        S = d[m]
        print(f"{lbl:<28s}{len(S):>5d}{S['oc'].gt(0).mean():>11.0%}{S['oc'].mean()*100:>+9.3f}%{S['green'].mean():>10.0%}")
    print("\n TIME OF SESSION HIGH / LOW by state:")
    for lbl, m in states:
        S = d[m]
        print(f"  {lbl:<26s} HIGH: {timeprofile(S,'t_hi')}")
        print(f"  {'':<26s} LOW:  {timeprofile(S,'t_lo')}")
    # the lead case detail: prev down + gap down, magnitude buckets
    print("\n LEAD CASE prev-DOWN + gap-DOWN by gap size:")
    D = d[(d.prev_dn) & (d.gap < 0)].copy()
    D["b"] = pd.cut(D.gap * 100, [-99, -1.0, -0.5, -0.2, 0], labels=["<-1%", "-1..-0.5%", "-0.5..-0.2%", "-0.2..0%"])
    for b, S in D.groupby("b", observed=True):
        if len(S) < 15:
            continue
        print(f"  gap {b:<12s} n={len(S):>4d}  P(O->C up) {S['oc'].gt(0).mean():.0%}  avg {S['oc'].mean()*100:+.3f}%  "
              f"P(green) {S['green'].mean():.0%}  P(touch prevC) {S['touch_pc'].mean():.0%}  "
              f"HIGH med {S['t_hi'].median():.0f}m  LOW med {S['t_lo'].median():.0f}m")
    # halves stability for the lead case
    D["half"] = [1 if str(x) < "2021-07-01" else 2 for x in D["day"]]
    for hf, S in D.groupby("half"):
        print(f"  half {hf}: n={len(S)}  P(O->C up) {S['oc'].gt(0).mean():.0%}  avg {S['oc'].mean()*100:+.3f}%")
