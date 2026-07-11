"""DRILL: volume-surge DOWNSIDE break from tight range. Half-split + put-convexity.

Event: tight BB (bottom tercile), close < prior 6-bar low, breakout-bar dvol>=2.5.
Hold to 60min or EOD. Favorable = downward move (long put intrinsic proxy).
Put score: a long ATM-ish put pays ~ max(0, downmove_at_exit) - premium (as % underlying).
We report E[max(0,fav_exit)] = the BREAKEVEN premium: you're +EV if you buy for less.
Also report the best-exit version (capture MFE, not just drift) as the convex upside.
SPY+QQQ 5m, halves + AM.
"""
import os
from collections import defaultdict, deque
from datetime import time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
LB = 6


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
    hist = defaultdict(lambda: deque(maxlen=30))
    dvol = np.full(len(df), np.nan)
    vals = df[["slot", "volume"]].values; days = df["day"].values
    cur = None; pend = []
    for i, (s, v) in enumerate(vals):
        if days[i] != cur:
            for ss, vv in pend:
                hist[ss].append(vv)
            pend = []; cur = days[i]
        hh = hist[s]
        if len(hh) >= 15:
            dvol[i] = v / np.median(hh)
        pend.append((s, v))
    df["dvol"] = dvol
    return df


def run(sym):
    df = load(sym)
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    bbw = df["bbw"].values; dvol = df["dvol"].values
    day = df["day"].values; ts = df["ts"].values; slot = df["slot"].values
    n = len(df)
    de_of = {}; r0_of = {}
    for i, dy in enumerate(day):
        de_of[dy] = i
        r0_of.setdefault(dy, i)
    WARM = np.datetime64("2016-06-01")
    q33 = np.nanpercentile(bbw[~np.isnan(bbw)], 33)

    ev = []
    for i in range(30, n-1):
        if ts[i] < WARM or np.isnan(bbw[i]) or np.isnan(dvol[i]):
            continue
        if i - LB < r0_of[day[i]] or bbw[i] > q33 or dvol[i] < 2.5:
            continue
        de = de_of[day[i]]
        if de - i < 4:
            continue
        pri_l = l[i-LB:i].min()
        if not (c[i] < pri_l):        # DOWN break only
            continue
        end = min(i+12, de)
        drift_dn = -(c[end]/c[i]-1)            # + = fell (favorable for put)
        mfe_dn = -(l[i+1:end+1].min()/c[i]-1)  # best downward excursion (convex capture)
        mae_up = (h[i+1:end+1].max()/c[i]-1)   # adverse pop
        ev.append(dict(drift_dn=drift_dn, mfe_dn=mfe_dn, mae_up=mae_up,
                       am=slot[i] < 690, half=1 if ts[i] < np.datetime64("2021-07-01") else 2))
    E = pd.DataFrame(ev)
    print(f"\n============ {sym}  surge downside breaks n={len(E)} ============")

    def rep(lbl, S):
        if len(S) < 30:
            print(f"  {lbl:<14s} n={len(S)} (thin)"); return
        d = S["drift_dn"].values; m = S["mfe_dn"].values
        pdn = (d > 0).mean()                       # % that closed lower at exit
        be_hold = np.maximum(d, 0).mean()          # breakeven premium, hold-to-exit
        be_mfe = np.maximum(m, 0).mean()           # breakeven if you capture the MFE
        print(f"  {lbl:<14s} n={len(S):>4d}  P(down@exit) {pdn:4.0%}  medMFEdn {np.median(m)*100:5.3f}%  "
              f"BE-prem hold {be_hold*100:5.3f}%  BE-prem bestexit {be_mfe*100:5.3f}%")
    rep("ALL", E)
    rep("half 1", E[E.half == 1])
    rep("half 2", E[E.half == 2])
    rep("AM", E[E.am])
    rep("half2 AM", E[(E.half == 2) & E.am])
    print("  (BE-prem = avg favorable move; a put is +EV only if it costs LESS than this,")
    print("   minus fees. 'bestexit' assumes you sell at the intraday low, an upper bound.)")
    return E


for sym in ["SPY", "QQQ"]:
    run(sym)
