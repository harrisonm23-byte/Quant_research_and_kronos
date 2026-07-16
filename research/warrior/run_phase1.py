"""Warrior pattern spec — Phase 1 signal utility, generic liquid track.

Detectors: their code, unmodified (warrior_pattern_backtest.py).
Data: SPY+QQQ 5m 2016-2026 (RTH), SPY 1m 2022-2026 (RTH) for the 10-candle rule.
Per spec: next-bar-open conservative execution AND their intrabar R-simulator,
random baseline matched by symbol/time-of-day, halves split, MFE/MAE, races.
"""
import os, sys
from datetime import time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "warrior_backtest"))
from warrior_pattern_backtest import (consecutive_candle_reversals, flat_top_breakouts,
                                      failed_flat_top_breakouts, simulate_intraday_barrier_trade)

DBT = "/tmp/claude-0/-home-user-Kronos/2f0190a1-7250-5bf8-81dd-e6806ae4a3ce/scratchpad/daily_bt"
NY = ZoneInfo("America/New_York")

# --- smoke test (their canonical case): 5 red then new high must fire long ---
idx = pd.date_range("2024-01-02 10:00", periods=8, freq="5min")
smoke = pd.DataFrame({
    "open":  [100, 99.5, 99, 98.5, 98, 97.5, 97.4, 97.8],
    "high":  [100.2, 99.6, 99.1, 98.6, 98.1, 97.6, 98.2, 98.5],
    "low":   [99.4, 98.9, 98.4, 97.9, 97.4, 97.0, 97.2, 97.6],
    "close": [99.5, 99, 98.5, 98, 97.5, 97.2, 98.0, 98.3],
}, index=idx)
s = consecutive_candle_reversals(smoke, n_consecutive=5)
assert any(x.side == "long" and x.signal_index == 6 for x in s), "smoke test failed"
print("smoke test OK (5-red reversal fires on canonical case)")


def load5(sym):
    df = pd.read_csv(os.path.join(DBT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))]
    df = df.sort_values("ts").set_index("ts")
    return df[["open", "high", "low", "close", "volume"]]


def load1():
    df = pd.read_csv(os.path.join(DBT, "SPY_1m.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 59))]
    df = df.sort_values("ts").set_index("ts")
    return df[["open", "high", "low", "close", "volume"]]


def phase1(df, signals, label, H=12, race_t=0.0025, per_day_dedup_bars=6):
    """Next-bar-open execution: enter open[i+1], measure fwd/MFE/MAE H bars, same-day only."""
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    days = df.index.date
    n = len(df)
    half_cut = pd.Timestamp("2021-07-01", tz=NY)
    rows = []
    last_by_side = {}
    for s in signals:
        i = s.signal_index
        if i + 1 >= n or days[i] != days[i + 1]:
            continue
        key = (s.pattern, s.side)
        if i - last_by_side.get(key, -999) < per_day_dedup_bars:
            continue
        last_by_side[key] = i
        e = o[i + 1]
        end = i + 1
        while end + 1 < n and days[end + 1] == days[i] and end + 1 <= i + 1 + H:
            end += 1
        if end <= i + 1:
            continue
        sgn = 1 if s.side == "long" else -1
        seg_h = h[i + 2:end + 1] if end >= i + 2 else h[i + 1:end + 1]
        seg_l = l[i + 2:end + 1] if end >= i + 2 else l[i + 1:end + 1]
        if len(seg_h) == 0:
            continue
        mfe = (seg_h.max() / e - 1) * sgn if sgn > 0 else (1 - seg_l.min() / e)
        mae = (1 - seg_l.min() / e) * -1 if sgn > 0 else (seg_h.max() / e - 1) * -1
        drift = (c[end] / e - 1) * sgn
        # race
        up = e * (1 + race_t); dn = e * (1 - race_t)
        race = 0
        for j in range(i + 2, end + 1):
            hu = h[j] >= up; du = l[j] <= dn
            if hu and du:
                break
            if hu:
                race = sgn; break
            if du:
                race = -sgn; break
        rows.append(dict(side=s.side, drift=drift, mfe=mfe, mae=mae, race=race,
                         half=1 if df.index[i] < half_cut else 2))
    E = pd.DataFrame(rows)
    if not len(E):
        print(f"  {label}: 0 events"); return None
    for side in E["side"].unique():
        S = E[E.side == side]
        w = (S.race == 1).sum(); lo_ = (S.race == -1).sum()
        h1 = S[S.half == 1]; h2 = S[S.half == 2]
        print(f"  {label:<34s} {side:<5s} n={len(S):>5d}  drift {S.drift.mean()*100:+.3f}%  "
              f"medMFE {S.mfe.median()*100:.3f}%  medMAE {S.mae.median()*100:+.3f}%  "
              f"race {w}/{lo_} ({w/max(w+lo_,1):.0%})  halves drift {h1.drift.mean()*100:+.3f}/{h2.drift.mean()*100:+.3f}%")
    return E


def baseline(df, label, H=12, race_t=0.0025):
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    days = df.index.date; n = len(df)
    drifts = []; races = []
    for i in range(30, n - H - 2, 97):
        if days[i] != days[i + 1]:
            continue
        e = o[i + 1]
        end = i + 1 + H
        if end >= n or days[end] != days[i]:
            continue
        drifts.append(c[end] / e - 1)
        up = e * (1 + race_t); dn = e * (1 - race_t); r = 0
        for j in range(i + 2, end + 1):
            hu = h[j] >= up; du = l[j] <= dn
            if hu and du: break
            if hu: r = 1; break
            if du: r = -1; break
        races.append(r)
    d = np.array(drifts); r = np.array(races)
    w = (r == 1).sum(); lo_ = (r == -1).sum()
    print(f"  {label:<34s} BASE  n={len(d):>5d}  drift {d.mean()*100:+.3f}%  race up {w/max(w+lo_,1):.0%}")


def rsim(df, signals, label):
    """Their intrabar R-simulator, guide defaults (1R stop / 2R target, 0.25% risk)."""
    res = []
    last = {}
    for s in signals:
        key = (s.pattern, s.side)
        if s.signal_index - last.get(key, -999) < 6:
            continue
        last[key] = s.signal_index
        try:
            r = simulate_intraday_barrier_trade(df, s)
        except Exception:
            continue
        res.append(r)
    if not res:
        return
    R = pd.DataFrame(res)
    for side in R["side"].unique():
        S = R[R.side == side]
        gains = S[S.pnl_pct > 0].pnl_pct.sum(); losses = -S[S.pnl_pct < 0].pnl_pct.sum()
        pf = gains / losses if losses > 0 else float("inf")
        print(f"  {label:<34s} {side:<5s} [their R-sim] n={len(S):>5d}  WR {(S.pnl_pct>0).mean():.0%}  "
              f"PF {pf:.2f}  avg {S.pnl_pct.mean()*100:+.3f}%  target/stop/time {sum(S.reason=='target')}/{sum(S.reason=='stop')}/{sum(S.reason=='time')}")


for sym in ["SPY", "QQQ"]:
    df = load5(sym)
    print(f"\n================ {sym} 5m (2016-2026, RTH) ================")
    baseline(df, f"{sym} 5m")
    sig_rev = consecutive_candle_reversals(df, n_consecutive=5)
    phase1(df, sig_rev, "5-candle exhaustion reversal")
    rsim(df, sig_rev, "5-candle exhaustion reversal")
    sig_ft = flat_top_breakouts(df, min_touches=3, tolerance_pct=0.001)
    phase1(df, sig_ft, "flat-top breakout (3 touch, 0.10%)")
    rsim(df, sig_ft, "flat-top breakout")
    sig_ff = failed_flat_top_breakouts(df, min_touches=3, tolerance_pct=0.001)
    phase1(df, sig_ff, "failed flat-top (bull trap short)")
    rsim(df, sig_ff, "failed flat-top (intrabar CAVEAT)")

print("\n================ SPY 1m (2022-2026, RTH) ================")
d1 = load1()
baseline(d1, "SPY 1m", H=24)
sig10 = consecutive_candle_reversals(d1, n_consecutive=10)
phase1(d1, sig10, "10-candle exhaustion reversal", H=24)
rsim(d1, sig10, "10-candle exhaustion reversal")
