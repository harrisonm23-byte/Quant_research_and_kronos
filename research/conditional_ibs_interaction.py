"""IBS interaction with conditional open states.

Does the down-down state sharpen IBS signals? If IBS<0.15 AND we're in a
consecutive-down + gap-down state, does the win rate go up?
Also: IBS on gap-fill days — does low IBS predict gap fill on gap-down days?
"""
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")


def load_daily(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"))
    df["ts"] = pd.to_datetime(df["date"])
    df = df.sort_values("ts").reset_index(drop=True)
    df["ibs"] = (df["close"] - df["low"]) / (df["high"] - df["low"])
    df["pc"] = df["close"].shift(1)
    df["gap"] = df["open"] / df["pc"] - 1
    df["fwd1"] = df["close"].shift(-1) / df["open"].shift(-1) - 1  # next day O→C
    df["fwd1_green"] = df["close"].shift(-1) > df["close"]  # next day closes above today's close
    df["prev_dn"] = df["close"] < df["close"].shift(1)
    df["prev2_dn"] = df["close"].shift(1) < df["close"].shift(2)
    df["prev3_dn"] = df["close"].shift(2) < df["close"].shift(3)
    df["oc"] = df["close"] / df["open"] - 1
    df["day_dn"] = df["close"] < df["open"]
    # next day gap
    df["fwd_gap"] = df["open"].shift(-1) / df["close"] - 1
    df["fwd_gap_dn"] = df["fwd_gap"] < 0
    return df.dropna(subset=["ibs", "fwd1", "prev_dn", "prev2_dn"])


for sym in ["SPY", "QQQ"]:
    d = load_daily(sym)
    print(f"\n{'='*70}")
    print(f" {sym} — IBS × Conditional States (n={len(d)})")
    print(f"{'='*70}")

    print(f"\n 1. IBS < 0.15 signal × context (next-day O→C)")
    print(f" {'context':<35s}{'n':>5s}{'P(+)':>6s}{'avg':>10s}{'med':>10s}")
    lo_ibs = d.ibs < 0.15
    for lbl, mask in [
        ("IBS<0.15 (baseline)",            lo_ibs),
        ("IBS<0.15 + prev down",           lo_ibs & d.prev_dn),
        ("IBS<0.15 + 2+ down",            lo_ibs & d.prev_dn & d.prev2_dn),
        ("IBS<0.15 + 3+ down",            lo_ibs & d.prev_dn & d.prev2_dn & d.prev3_dn),
        ("IBS<0.15 + gap down today",      lo_ibs & (d.gap < 0)),
        ("IBS<0.15 + down + gap dn",       lo_ibs & d.prev_dn & (d.gap < 0)),
        ("IBS<0.15 + 2+down + gap dn",    lo_ibs & d.prev_dn & d.prev2_dn & (d.gap < 0)),
        ("IBS<0.15 + today down (O→C)",    lo_ibs & d.day_dn),
        ("IBS<0.15 + gap dn + day dn",    lo_ibs & (d.gap < 0) & d.day_dn),
    ]:
        S = d[mask].dropna(subset=["fwd1"])
        if len(S) < 15:
            continue
        print(f" {lbl:<35s}{len(S):>5d}{S.fwd1.gt(0).mean():>6.0%}"
              f"{S.fwd1.mean()*100:>+9.3f}%{S.fwd1.median()*100:>+9.3f}%")

    print(f"\n 2. IBS < 0.15 × next-day gap direction → next-day O→C")
    for lbl, mask in [
        ("IBS<0.15 + next gap down",   lo_ibs & d.fwd_gap_dn),
        ("IBS<0.15 + next gap up",     lo_ibs & (~d.fwd_gap_dn)),
    ]:
        S = d[mask].dropna(subset=["fwd1"])
        if len(S) < 15:
            continue
        print(f" {lbl:<35s}{len(S):>5d}{S.fwd1.gt(0).mean():>6.0%}"
              f"{S.fwd1.mean()*100:>+9.3f}%{S.fwd1.median()*100:>+9.3f}%")

    # IBS at various thresholds combined with down context
    print(f"\n 3. IBS threshold sweep × 'prev down + gap down today'")
    ctx = d.prev_dn & (d.gap < 0)
    print(f" {'IBS threshold':<20s}{'n':>5s}{'P(+)':>6s}{'avg':>10s}")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        S = d[ctx & (d.ibs < thr)].dropna(subset=["fwd1"])
        if len(S) < 15:
            continue
        print(f" IBS < {thr:<13.2f}{len(S):>5d}{S.fwd1.gt(0).mean():>6.0%}"
              f"{S.fwd1.mean()*100:>+9.3f}%")

    # Without the down context (baseline)
    print(f" (baseline without context:)")
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        S = d[d.ibs < thr].dropna(subset=["fwd1"])
        if len(S) < 15:
            continue
        print(f" IBS < {thr:<13.2f}{len(S):>5d}{S.fwd1.gt(0).mean():>6.0%}"
              f"{S.fwd1.mean()*100:>+9.3f}%")

    # Halves
    print(f"\n 4. HALVES: IBS<0.15 × context")
    d["half"] = [1 if str(x) < "2021" else 2 for x in d["ts"].dt.year]
    print(f" {'state':<35s}{'half':>5s}{'n':>5s}{'P(+)':>6s}{'avg':>10s}")
    for lbl, mask in [
        ("IBS<0.15 baseline",           lo_ibs),
        ("IBS<0.15 + prev dn + gap dn", lo_ibs & d.prev_dn & (d.gap < 0)),
        ("IBS<0.15 + 2+dn + gap dn",   lo_ibs & d.prev_dn & d.prev2_dn & (d.gap < 0)),
    ]:
        for hf in [1, 2]:
            S = d[mask & (d.half == hf)].dropna(subset=["fwd1"])
            if len(S) < 10:
                continue
            print(f" {lbl:<35s}{hf:>5d}{len(S):>5d}{S.fwd1.gt(0).mean():>6.0%}"
                  f"{S.fwd1.mean()*100:>+9.3f}%")
