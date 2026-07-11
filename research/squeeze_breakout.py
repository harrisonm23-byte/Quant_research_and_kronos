"""DISCOVERY: volume-surge breakout from a TIGHT consolidation (the user's model).

Consolidation: BB width in bottom tercile (tight) for the setup bar, AND the last
LB=6 bars held a narrow range (range <= NR x ATR-ish). Breakout bar: close pushes
beyond the last LB-bar high (UP) or low (DOWN). Volume trigger: deseasonalized
volume of the breakout bar. We MAP forward continuation in the BREAK DIRECTION as a
function of volume surge, and compare volume-surge breakouts vs quiet breakouts.

Question answered: does a volume increase determine the price change (directional,
sustained) out of a range? Race +T vs -T signed to break dir; MFE/MAE/drift.
SPY+QQQ 5m, both halves shown. Winners uncapped.
"""
import os
from collections import defaultdict, deque
from datetime import time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
LB = 6   # consolidation lookback (30 min)


def load(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
    df["day"] = df["ts"].dt.date
    df["slot"] = df["ts"].dt.hour*60 + df["ts"].dt.minute
    c = df["close"]
    df["sma20"] = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    df["bbw"] = (4*sd) / df["sma20"]
    # deseasonalized volume: vol / trailing-30-session median for that slot
    hist = defaultdict(lambda: deque(maxlen=30))
    dvol = np.full(len(df), np.nan)
    vals = df[["slot", "volume"]].values
    days = df["day"].values
    cur = None; pend = []
    for i, (s, v) in enumerate(vals):
        if days[i] != cur:
            for ss, vv in pend:
                hist[ss].append(vv)
            pend = []; cur = days[i]
        h = hist[s]
        if len(h) >= 15:
            dvol[i] = v / np.median(h)
        pend.append((s, v))
    df["dvol"] = dvol
    return df


def run(sym):
    df = load(sym)
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    bbw = df["bbw"].values; dvol = df["dvol"].values
    day = df["day"].values; ts = df["ts"].values
    slot = df["slot"].values
    n = len(df)
    de_of = {}
    for i, dy in enumerate(day):
        de_of[dy] = i
    r0_of = {}
    for i, dy in enumerate(day):
        r0_of.setdefault(dy, i)
    WARM = np.datetime64("2016-06-01")
    q33 = np.nanpercentile(bbw[~np.isnan(bbw)], 33)

    def race(i, T, sgn, H=12):
        up = c[i]*(1+T); dn = c[i]*(1-T)
        end = min(i+H, de_of[day[i]])
        for j in range(i+1, end+1):
            hu = h[j] >= up; du = l[j] <= dn
            if hu and du:
                return 0
            if hu:
                return 1*sgn
            if du:
                return -1*sgn
        return 0

    ev = []
    for i in range(30, n-1):
        if ts[i] < WARM or np.isnan(bbw[i]) or np.isnan(dvol[i]):
            continue
        r0 = r0_of[day[i]]
        if i - LB < r0:                       # need LB bars same day before
            continue
        de = de_of[day[i]]
        if de - i < 4:
            continue
        if bbw[i] > q33:                       # require TIGHT consolidation
            continue
        pri_h = h[i-LB:i].max(); pri_l = l[i-LB:i].min()
        up = c[i] > pri_h
        dn = c[i] < pri_l
        if not (up or dn):
            continue
        sgn = 1 if up else -1
        end = min(i+12, de)
        mfe = (h[i+1:end+1].max()/c[i]-1) * sgn    # favorable = break dir
        mae = (l[i+1:end+1].min()/c[i]-1) * sgn
        if sgn < 0:
            mfe = (c[i]/l[i+1:end+1].min()-1)       # recompute favorable for down
            mae = (c[i]/h[i+1:end+1].max()-1)
        drift = (c[end]/c[i]-1) * sgn
        ev.append(dict(dir="UP" if up else "DOWN", dvol=dvol[i], drift=drift,
                       mfe=mfe, mae=mae, r2=race(i, 0.002, sgn), r3=race(i, 0.003, sgn),
                       am=slot[i] < 690,
                       half=1 if ts[i] < np.datetime64("2021-07-01") else 2))
    E = pd.DataFrame(ev)
    print(f"\n================= {sym}  (tight-range breakouts n={len(E)}) =================")

    def rep(lbl, S):
        if len(S) < 40:
            print(f"  {lbl:<28s} n={len(S)} (thin)"); return
        d = S["drift"].values; m = S["mfe"].values; a = S["mae"].values
        w3 = (S["r3"] == 1).sum(); l3 = (S["r3"] == -1).sum()
        print(f"  {lbl:<28s} n={len(S):>4d}  contin(drift) {d.mean()*100:+5.3f}%  favMFE {np.median(m)*100:5.3f}%  "
              f"advMAE {np.median(a)*100:5.3f}%  race0.3 {w3}/{l3} ({w3/max(w3+l3,1):.0%} cont)")

    print(" volume surge of the breakout bar (deseasonalized):")
    rep("QUIET break dvol<1.0", E[E.dvol < 1.0])
    rep("normal 1.0-1.5", E[(E.dvol >= 1.0) & (E.dvol < 1.5)])
    rep("elevated 1.5-2.5", E[(E.dvol >= 1.5) & (E.dvol < 2.5)])
    rep("SURGE 2.5-4", E[(E.dvol >= 2.5) & (E.dvol < 4)])
    rep("BIG SURGE >=4", E[E.dvol >= 4])
    print(" surge >=2.5 breakdown:")
    S = E[E.dvol >= 2.5]
    rep("surge ALL", S)
    rep("surge UP", S[S.dir == "UP"])
    rep("surge DOWN", S[S.dir == "DOWN"])
    rep("surge AM", S[S.am])
    rep("surge half1", S[S.half == 1])
    rep("surge half2", S[S.half == 2])
    return E


for sym in ["SPY", "QQQ"]:
    run(sym)
