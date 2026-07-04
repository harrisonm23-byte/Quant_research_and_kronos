"""Overbought SHORT test on SPY across timeframes.

Signal (at close of bar t):
  1. close > VWAP        (daily: 20-day rolling VWAP; intraday: session VWAP)
  2. RSI14 > {70, 80}
  3. close > upper Bollinger Band (20, 2.0)  [also reported: %B > 0.95]
Short entry at next bar open; binary win = price BELOW entry at horizon close.
Reported vs baseline (short any bar) so the drift headwind is visible.
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")


def wilder_rsi(close, period):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def add_common(df):
    c = df["close"]
    df["rsi14"] = wilder_rsi(c, 14)
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    df["bb_up"] = mid + 2 * sd
    df["pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
    return df


# ---------- daily ----------
daily = pd.read_csv(os.path.join(OUT, "SPY_daily.csv"), parse_dates=["date"])
tp = (daily["high"] + daily["low"] + daily["close"]) / 3
daily["vwap20"] = (tp * daily["volume"]).rolling(20).sum() / daily["volume"].rolling(20).sum()
daily = add_common(daily)
daily["above_vwap"] = daily["close"] > daily["vwap20"]

# ---------- intraday ----------
intra = pd.read_csv(os.path.join(OUT, "SPY_30m_full.csv"))
intra["ts"] = pd.to_datetime(intra["timestamps"]).dt.tz_convert(NY)
keep = (intra["ts"].dt.time >= dtime(9, 30)) & (intra["ts"].dt.time <= dtime(15, 30))
intra = intra[keep].sort_values("ts").reset_index(drop=True)
# session VWAP: cumulative per day using per-bar vwap * volume
intra["day"] = intra["ts"].dt.date
pv = intra["vwap"] * intra["volume"]
intra["svwap"] = pv.groupby(intra["day"]).cumsum() / intra["volume"].groupby(intra["day"]).cumsum()


def agg_1h(df30):
    minutes = (df30["ts"].dt.hour * 60 + df30["ts"].dt.minute) - (9 * 60 + 30)
    grp = np.minimum(minutes // 60, 6)
    key = df30["ts"].dt.date.astype(str) + "_" + grp.astype(str)
    g = df30.groupby(key, sort=False)
    out = pd.DataFrame({
        "ts": g["ts"].first(), "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum(),
        "svwap": g["svwap"].last(),
    }).reset_index(drop=True).sort_values("ts").reset_index(drop=True)
    return out


h1 = agg_1h(intra)
intra = add_common(intra)
h1 = add_common(h1)
intra["above_vwap"] = intra["close"] > intra["svwap"]
h1["above_vwap"] = h1["close"] > h1["svwap"]

WARM = pd.Timestamp("2016-06-01")
CFGS = [
    ("daily", daily, "date", [1, 2, 3, 5, 10], "sessions"),
    ("1h", h1, "ts", [1, 2, 4, 7, 14, 35], "bars"),
    ("30m", intra, "ts", [2, 4, 8, 13, 26, 65], "bars"),
]


def horizon_wr(df, tcol, sig_mask, horizons):
    o = df["open"].values
    c = df["close"].values
    n = len(df)
    idxs = np.flatnonzero(sig_mask.fillna(False).values)
    out = {}
    for h in horizons:
        rets = []
        for i in idxs:
            if i + h < n:
                entry = o[i + 1] * (1 - 0.0002)      # short fill at next open
                rets.append(entry / c[i + h] - 1)    # short return
        a = np.array(rets)
        out[h] = (len(a), (a > 0).mean() if len(a) else float("nan"),
                  a.mean() if len(a) else float("nan"))
    return out


for name, df, tcol, horizons, unit in CFGS:
    t = df[tcol]
    if hasattr(t.dt, "tz_localize"):
        pass
    warm_mask = pd.to_datetime(t).dt.tz_localize(None) >= WARM if t.dt.tz is not None else t >= WARM
    base = warm_mask & df["rsi14"].notna() & df["bb_up"].notna()
    print(f"\n=== {name} (horizons in {unit}) ===")
    variants = [
        ("BASELINE (any bar)", base),
        ("VWAP+RSI70+BBupper", base & df["above_vwap"] & (df["rsi14"] > 70) & (df["close"] > df["bb_up"])),
        ("VWAP+RSI70+%B>.95", base & df["above_vwap"] & (df["rsi14"] > 70) & (df["pctb"] > 0.95)),
        ("VWAP+RSI80+BBupper", base & df["above_vwap"] & (df["rsi14"] > 80) & (df["close"] > df["bb_up"])),
        ("RSI70 only", base & (df["rsi14"] > 70)),
    ]
    hdr = f"{'variant':<20s} " + " ".join(f"{'h='+str(h):>14s}" for h in horizons)
    print(hdr)
    for vname, mask in variants:
        st = horizon_wr(df, tcol, mask, horizons)
        cells = []
        for h in horizons:
            n, wr, avg = st[h]
            cells.append(f"{wr:>5.1%}/{avg:>+6.2%}({n})" if n else "      n=0     ")
        print(f"{vname:<20s} " + " ".join(f"{cell:>14s}" for cell in cells))
print("\ncell = shortWR / avg short return (n signals); win = price below entry at horizon")
