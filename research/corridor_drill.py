"""DRILL the one positive-skew basin: long coil (dwell>=11 in corridor) + WIDE bands.
De-overlap: EVENT = the bar where dwell first reaches 11 while bbw is wide (>q67).
Then race forward 12 bars (60min): does +T hit before -T? Characterize the move.
Split both halves x both symbols. Also a convex-payoff proxy (defined-risk long).
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
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
    df["day"] = df["ts"].dt.date
    c = df["close"]
    df["sma9"] = c.rolling(9).mean()
    df["sma20"] = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    df["lbb"] = df["sma20"] - 2*sd
    df["ubb"] = df["sma20"] + 2*sd
    df["bbw"] = (df["ubb"] - df["lbb"]) / df["sma20"]
    return df


def race(i, c, h, l, de, T, H=12):
    up = c[i]*(1+T); dn = c[i]*(1-T)
    end = min(i+H, de)
    for j in range(i+1, end+1):
        hu = h[j] >= up; du = l[j] <= dn
        if hu and du:
            return 0  # ambiguous same-bar
        if hu:
            return 1
        if du:
            return -1
    return 0


def run(sym):
    df = load(sym)
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    s9 = df["sma9"].values; lbb = df["lbb"].values; bbw = df["bbw"].values
    day = df["day"].values; ts = df["ts"].values
    tod = (df["ts"].dt.hour*60 + df["ts"].dt.minute).values
    n = len(df)
    de_of = {}
    for i, dy in enumerate(day):
        de_of[dy] = i
    WARM = np.datetime64("2016-06-01")
    q67 = np.nanpercentile(bbw[~np.isnan(bbw)], 67)

    in_zone = (~np.isnan(s9)) & (~np.isnan(lbb)) & (s9 > lbb) & (c < s9) & (c > lbb)
    dwell = np.zeros(n, int)
    for i in range(1, n):
        if day[i] == day[i-1] and in_zone[i]:
            dwell[i] = dwell[i-1]+1 if in_zone[i-1] else 1
        elif in_zone[i]:
            dwell[i] = 1

    ev = []
    for i in range(30, n-1):
        if ts[i] < WARM:
            continue
        # de-overlapped entry: dwell crosses to >=11 (was <11 prior bar), wide bands
        if in_zone[i] and dwell[i] >= 11 and dwell[i-1] < 11 and bbw[i] > q67:
            de = de_of[day[i]]
            if de - i < 4:
                continue
            end = min(i+12, de)
            mfe = h[i+1:end+1].max()/c[i]-1
            mae = l[i+1:end+1].min()/c[i]-1
            drift = c[end]/c[i]-1
            r2 = race(i, c, h, l, de, 0.002)
            r3 = race(i, c, h, l, de, 0.003)
            ev.append(dict(mfe=mfe, mae=mae, drift=drift, r2=r2, r3=r3,
                           am=tod[i] < 690,
                           half=1 if ts[i] < np.datetime64("2021-07-01") else 2))
    E = pd.DataFrame(ev)
    print(f"\n===== {sym}  (de-overlapped long-coil+wide events: n={len(E)}) =====")

    def rep(lbl, S):
        if len(S) < 20:
            print(f"  {lbl:<16s} n={len(S)} (thin)"); return
        m = S["mfe"].values; a = S["mae"].values; d = S["drift"].values
        r2 = S["r2"].values; r3 = S["r3"].values
        w2 = (r2 == 1).sum(); l2 = (r2 == -1).sum()
        w3 = (r3 == 1).sum(); l3 = (r3 == -1).sum()
        print(f"  {lbl:<16s} n={len(S):>4d}  MFE {np.median(m)*100:5.3f}%  MAE {np.median(a)*100:6.3f}%  "
              f"drift {d.mean()*100:+5.3f}%  race0.2 {w2}/{l2} ({w2/max(w2+l2,1):.0%}up)  "
              f"race0.3 {w3}/{l3} ({w3/max(w3+l3,1):.0%}up)")
    rep("ALL", E)
    rep("half 1", E[E.half == 1])
    rep("half 2", E[E.half == 2])
    rep("AM only", E[E.am])
    rep("half1 AM", E[(E.half == 1) & E.am])
    rep("half2 AM", E[(E.half == 2) & E.am])
    return E


for sym in ["SPY", "QQQ"]:
    run(sym)
