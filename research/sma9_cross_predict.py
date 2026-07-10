"""Predicting SMA9 crosses on 5m (SPY+QQQ, 2016-2026).

Event: bar i closes with an established up-run (>=4 consecutive closes > SMA9).
Question 1: P(bar i+1 CLOSES below SMA9)? Which features at bar i move it?
Question 2: economics — after a confirmed cross vs a touch-and-hold, what does
price do over the next 30/60 min? (Does the cross even matter?)
Mirror run for down-runs (predicting upward crosses).
"""
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")


def wilder_rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    au = np.zeros_like(c); ad = np.zeros_like(c)
    au[n] = up[1:n+1].mean(); ad[n] = dn[1:n+1].mean()
    for i in range(n+1, len(c)):
        au[i] = (au[i-1]*(n-1) + up[i]) / n
        ad[i] = (ad[i-1]*(n-1) + dn[i]) / n
    rs = np.divide(au, ad, out=np.full_like(c, np.inf), where=ad > 0)
    return 100 - 100/(1+rs)


def load(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
    df["day"] = df["ts"].dt.date
    c = df["close"]
    df["sma9"] = c.rolling(9).mean()
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["sma20"] = c.rolling(20).mean()
    df["volr"] = df["volume"] / df["volume"].rolling(20).mean()
    df["rsi"] = wilder_rsi(c.values)
    pv = df["vwap"] * df["volume"]
    df["svwap"] = pv.groupby(df["day"]).cumsum() / df["volume"].groupby(df["day"]).cumsum()
    return df


def bucket_table(E, feat, edges, labels, outcome="cross"):
    rows = []
    for lo, hi, lb in zip(edges[:-1], edges[1:], labels):
        sub = E[(E[feat] >= lo) & (E[feat] < hi)]
        if len(sub) >= 200:
            rows.append((lb, len(sub), sub[outcome].mean()))
    return rows


def run(sym):
    df = load(sym)
    c = df["close"].values; l = df["low"].values; h = df["high"].values
    s9 = df["sma9"].values; e9 = df["ema9"].values; s20 = df["sma20"].values
    vr = df["volr"].values; rsi = df["rsi"].values; vw = df["svwap"].values
    day = df["day"].values
    n = len(df)
    rows_of = {}
    for i, dy in enumerate(day):
        rows_of.setdefault(dy, []).append(i)
    WARM = np.datetime64("2016-06-01")
    ts = df["ts"].values

    # consecutive closes above / below sma9
    above = np.zeros(n, int); below = np.zeros(n, int)
    for i in range(1, n):
        if day[i] != day[i-1]:
            continue  # runs reset across days (leave 0)
        if not np.isnan(s9[i]):
            above[i] = above[i-1] + 1 if c[i] > s9[i] else 0
            below[i] = below[i-1] + 1 if c[i] < s9[i] else 0

    ev = []
    for i in range(30, n - 1):
        if ts[i] < WARM or day[i] != day[i+1]:
            continue
        r0 = rows_of[day[i]][0]
        if i - r0 < 9:      # need sma9 fully intraday-ish
            continue
        de = rows_of[day[i]][-1]
        if above[i] >= 4:
            side = "up"
        elif below[i] >= 4:
            side = "down"
        else:
            continue
        px = c[i]
        cross = (c[i+1] < s9[i+1]) if side == "up" else (c[i+1] > s9[i+1])
        touch = (l[i+1] <= s9[i+1]) if side == "up" else (h[i+1] >= s9[i+1])
        # forward returns after bar i+1 close (30/60 min), signed favorable to trend
        f6 = f12 = np.nan
        if i + 1 + 6 <= de:
            f6 = (c[i+7] / c[i+1] - 1) * (1 if side == "up" else -1)
        if i + 1 + 12 <= de:
            f12 = (c[i+13] / c[i+1] - 1) * (1 if side == "up" else -1)
        sgn = 1 if side == "up" else -1
        ev.append(dict(
            side=side, cross=cross, touch_hold=(touch and not cross),
            dist=sgn*(px - s9[i]) / px * 100,           # % close beyond sma9 (signed to trend)
            ema_lead=sgn*(e9[i] - s9[i]) / px * 100,     # ema9 vs sma9, + = with trend
            slope9=sgn*(s9[i] - s9[i-3]) / px * 100,     # sma9 3-bar slope, trendward
            volr=vr[i], rsi=rsi[i] if side == "up" else 100-rsi[i],
            vwapd=sgn*(px - vw[i]) / px * 100,
            run=min(above[i] if side == "up" else below[i], 40),
            red=int(sgn*(c[i]-c[i-1]) < 0) + int(sgn*(c[i-1]-c[i-2]) < 0),  # counter-trend closes, last 2
            f6=f6, f12=f12, half=1 if ts[i] < np.datetime64("2021-07-01") else 2,
        ))
    E = pd.DataFrame(ev)
    print(f"\n================ {sym} ================")
    for side in ["up", "down"]:
        S = E[E.side == side]
        base = S["cross"].mean()
        print(f"\n--- {side}-run (n={len(S):,}), base P(cross next bar) = {base:.1%} ---")
        specs = [
            ("ema_lead", [-9, -0.02, 0, 0.02, 0.05, 9], ["ema OPPOSED <-0.02%", "ema -0.02..0", "ema 0..+0.02", "ema +0.02..0.05", "ema >+0.05%"]),
            ("dist",     [0, 0.03, 0.08, 0.15, 9],      ["close 0-0.03% past", "0.03-0.08%", "0.08-0.15%", ">0.15%"]),
            ("slope9",   [-9, 0, 0.02, 0.05, 9],        ["slope AGAINST", "slope 0..0.02", "0.02..0.05", ">0.05 strong"]),
            ("volr",     [0, 0.8, 1.2, 1.8, 99],        ["quiet <0.8x", "normal", "loud 1.2-1.8x", "very loud >1.8x"]),
            ("rsi",      [0, 50, 60, 70, 101],          ["rsi<50 (trendward)", "50-60", "60-70", ">70"]),
            ("red",      [0, 1, 2, 3],                  ["0 counter closes", "1 counter", "2 counter"]),
            ("run",      [4, 8, 16, 99],                ["run 4-7 bars", "8-15", "16+"]),
        ]
        for feat, edges, labels in specs:
            rows = bucket_table(S, feat, edges, labels)
            if not rows:
                continue
            line = " | ".join(f"{lb}: {p:.0%} (n={cnt//1000}k)" if cnt >= 2000 else f"{lb}: {p:.0%} ({cnt})" for lb, cnt, p in rows)
            print(f"  {feat:<9s} {line}")
        # combo cell: ema opposed + >=2 counter closes + quiet-vs-loud
        combo = S[(S.ema_lead < 0) & (S.red >= 2)]
        if len(combo) > 300:
            print(f"  COMBO ema-opposed & 2 counter closes: P(cross)={combo['cross'].mean():.0%} (n={len(combo):,})"
                  f"  [halves: {combo[combo.half==1]['cross'].mean():.0%}/{combo[combo.half==2]['cross'].mean():.0%}]")
        safe = S[(S.ema_lead > 0.02) & (S.red == 0)]
        if len(safe) > 300:
            print(f"  COMBO ema-leading & 0 counter closes: P(cross)={safe['cross'].mean():.0%} (n={len(safe):,})")
        # economics: what happens AFTER
        for lbl, sub in [("after CROSS", S[S.cross]), ("after TOUCH-HOLD", S[S.touch_hold]),
                         ("after clean hold (no touch)", S[~S.cross & ~S.touch_hold])]:
            m6 = sub["f6"].dropna(); m12 = sub["f12"].dropna()
            if len(m6) > 300:
                print(f"  {lbl:<28s} fwd30m {m6.mean()*100:+.3f}% (P>0 {(m6>0).mean():.0%})  fwd60m {m12.mean()*100:+.3f}%  n={len(m6):,}")
    return E


for sym in ["SPY", "QQQ"]:
    run(sym)
