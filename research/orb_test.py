"""ORB (opening range breakout) exactly as the video describes:
15-min opening range; first 5m CLOSE beyond the range = signal; enter next
bar open; stop = opposite side of range; TP at 1R / 2R / none; always flat
at EOD. One trade per day (first breakout). Costs 0.02%/side.
Variants: long / short / both; volume filter on breakout bar.
"""
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.0002


def load(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df.sort_values("ts").reset_index(drop=True)
    d5 = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].reset_index(drop=True)
    d5["day"] = d5["ts"].dt.date
    d5["vol20"] = d5["volume"].rolling(20).mean()
    return d5


def run_orb(d5, direction, tp_r, vol_filter=False):
    trades = []
    for dy, g in d5.groupby("day", sort=True):
        g = g.reset_index(drop=True)
        if len(g) < 20:
            continue
        rng_hi = g["high"].iloc[0:3].max()
        rng_lo = g["low"].iloc[0:3].min()
        rng = rng_hi - rng_lo
        if rng <= 0:
            continue
        side = None
        for k in range(3, len(g) - 1):
            if g["close"].iloc[k] > rng_hi and direction in ("long", "both"):
                side, sig_k = +1, k
                break
            if g["close"].iloc[k] < rng_lo and direction in ("short", "both"):
                side, sig_k = -1, k
                break
        if side is None:
            continue
        if vol_filter:
            va = g["vol20"].iloc[sig_k]
            if not va or g["volume"].iloc[sig_k] < 1.5 * va:
                continue
        e = g["open"].iloc[sig_k + 1] * (1 + SLIP * side)
        stop = rng_lo if side > 0 else rng_hi
        risk = abs(e - stop)
        if risk <= 0:
            continue
        tp = e + side * tp_r * risk if tp_r else None
        ret = None
        for k in range(sig_k + 1, len(g)):
            lo_, hi_ = g["low"].iloc[k], g["high"].iloc[k]
            if side > 0:
                if lo_ <= stop:
                    ret = (stop * (1 - SLIP)) / e - 1
                    break
                if tp and hi_ >= tp:
                    ret = (tp * (1 - SLIP)) / e - 1
                    break
            else:
                if hi_ >= stop:
                    ret = -((stop * (1 + SLIP)) / e - 1)
                    break
                if tp and lo_ <= tp:
                    ret = -((tp * (1 + SLIP)) / e - 1)
                    break
        if ret is None:
            x = g["close"].iloc[-1]
            ret = side * ((x * (1 - SLIP * side)) / e - 1)
        trades.append(ret)
    return np.array(trades)


for sym in ["SPY", "QQQ"]:
    d5 = load(sym)
    d5 = d5[d5["ts"] >= pd.Timestamp("2016-06-01", tz=NY)]
    years = (d5["ts"].iloc[-1] - d5["ts"].iloc[0]).days / 365.25
    print(f"===== {sym} ORB (15-min range, close-confirmed, stop=far side, EOD flat) =====")
    print(f"{'config':<26s}{'n':>6s}{'WR%':>7s}{'avg%':>8s}{'PF':>6s}{'ann%':>7s}{'maxDD%':>8s}")
    for direction in ["long", "short", "both"]:
        for tp_r in [1, 2, None]:
            a = run_orb(d5, direction, tp_r)
            if not len(a):
                continue
            eq = np.cumprod(1 + a)
            ann = eq[-1] ** (1 / years) - 1
            peak = np.maximum.accumulate(eq)
            mdd = ((eq - peak) / peak).min()
            wins, losses = a[a > 0], a[a <= 0]
            pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
            lbl = f"{direction}, TP={'EOD' if tp_r is None else str(tp_r)+'R'}"
            print(f"{lbl:<26s}{len(a):>6d}{(a>0).mean()*100:>7.1f}{a.mean()*100:>8.3f}"
                  f"{pf:>6.2f}{ann*100:>7.1f}{mdd*100:>8.1f}")
    # their "strong breakout" tease: volume filter on best-guess config
    a = run_orb(d5, "both", 2, vol_filter=True)
    if len(a):
        eq = np.cumprod(1 + a)
        ann = eq[-1] ** (1 / years) - 1
        peak = np.maximum.accumulate(eq)
        mdd = ((eq - peak) / peak).min()
        wins, losses = a[a > 0], a[a <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
        print(f"{'both, TP=2R, vol>1.5x':<26s}{len(a):>6d}{(a>0).mean()*100:>7.1f}"
              f"{a.mean()*100:>8.3f}{pf:>6.2f}{ann*100:>7.1f}{mdd*100:>8.1f}")
    print()
