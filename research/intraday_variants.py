"""Task 2: winners (Double Seven, IBS, Five-Day-Low) on QQQ 30m and 1h bars.

RTH bars only, next-bar-open fills, 0.02%/side slippage, positions may hold
overnight (consistent with the daily versions). 1h bars aggregated from 30m
starting at 09:30 (last bar of day is the 15:30-16:00 half hour).
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine import run_bt, compute_stats

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
STAT_START = pd.Timestamp("2016-06-01")   # warmup: 200 bars = ~15 sessions (30m)


def load_30m():
    df = pd.read_csv(os.path.join(OUT, "QQQ_30m_full.csv"))
    ts = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df["ts"] = ts
    keep = (ts.dt.time >= dtime(9, 30)) & (ts.dt.time <= dtime(15, 30))
    df = df[keep].sort_values("ts").reset_index(drop=True)
    return df


def agg_1h(df30):
    # group: 9:30-10:30, ..., 14:30-15:30, 15:30-16:00
    t = df30["ts"]
    minutes = (t.dt.hour * 60 + t.dt.minute) - (9 * 60 + 30)
    grp = np.minimum(minutes // 60, 6)
    key = t.dt.date.astype(str) + "_" + grp.astype(str)
    g = df30.groupby(key, sort=False)
    out = pd.DataFrame({
        "ts": g["ts"].first(), "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "volume": g["volume"].sum(),
    }).reset_index(drop=True).sort_values("ts").reset_index(drop=True)
    return out


def add_ind(df):
    c, h, l = df["close"], df["high"], df["low"]
    df["date"] = df["ts"].dt.tz_localize(None)   # engine uses 'date'
    df["sma200"] = c.rolling(200).mean()
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    df["lc7"] = c.rolling(7).min()
    df["hc7"] = c.rolling(7).max()
    df["lc5"] = c.rolling(5).min()
    df["prev_close"] = c.shift(1)
    return df


def nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


BARS = {"30m": add_ind(load_30m()), "1h": add_ind(agg_1h(load_30m()))}
BPD = {"30m": 13, "1h": 7}   # bars per session

RUNS = []
for tf in ["30m", "1h"]:
    RUNS.append((f"D7_{tf}", tf, dict(
        entry_fn=lambda r: nn(r.sma200, r.lc7) and r.close > r.sma200 and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7)))
    RUNS.append((f"D7_{tf}_nofilter", tf, dict(
        entry_fn=lambda r: nn(r.lc7) and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7)))
    for lo, hi in [(0.20, 0.70), (0.25, 0.75), (0.10, 0.80)]:
        RUNS.append((f"IBS_{tf}_e{int(lo*100)}_x{int(hi*100)}", tf, dict(
            entry_fn=(lambda lo_: lambda r: r.ibs < lo_)(lo),
            exit_fn=(lambda hi_: lambda r: r.ibs > hi_)(hi))))
    RUNS.append((f"5BarLow_IBS_{tf}", tf, dict(
        entry_fn=lambda r: nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)))

print(f"{'run':<22s} {'CAGR':>7s} {'maxDD':>7s} {'Sharpe':>6s} {'WR':>6s} {'PF':>5s} "
      f"{'#tr':>6s} {'avgtr':>8s} {'avghold':>8s}")
summary = []
for run_id, tf, kw in RUNS:
    df = BARS[tf]
    eq, trades = run_bt(df, stat_start=STAT_START, **kw)
    # Sharpe: eq is per-bar; annualize by bars/yr
    days = (eq.index[-1] - eq.index[0]).days
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total) ** (365.25 / days) - 1
    dr = eq.pct_change().dropna()
    ann_bars = BPD[tf] * 252
    sharpe = dr.mean() / dr.std() * math.sqrt(ann_bars) if dr.std() > 0 else 0
    peak = eq.cummax()
    maxdd = ((eq - peak) / peak).min()
    n = len(trades)
    if n:
        rets = trades["ret"].values
        wins, losses = rets[rets > 0], rets[rets <= 0]
        wr = len(wins) / n
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        avg = rets.mean()
        ah = trades["hold_days"].mean()   # in bars
    else:
        wr = pf = avg = ah = 0
    print(f"{run_id:<22s} {cagr:>7.1%} {maxdd:>7.1%} {sharpe:>6.2f} {wr:>6.1%} {pf:>5.2f} "
          f"{n:>6d} {avg:>8.3%} {ah:>6.1f}b")
    if n:
        trades.to_csv(os.path.join(OUT, f"trades_INTRA_{run_id}.csv"), index=False)
    summary.append((run_id, cagr, wr, pf, n))
print("\navghold in bars; window 2016-06 -> 2026-07; slippage 0.02%/side per trade")
