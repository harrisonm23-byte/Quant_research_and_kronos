"""Conditional open v2: sharpeners for the prev-DOWN + gap-DOWN recovery signal.

Tests two questions on SPY+QQQ 5m bars (2016-2026):
1. FIRST-30-MIN DIRECTION: on down-down days, does a first-half-hour flush (price
   drops further) vs immediate bounce (price rises) meaningfully split the 57% base?
2. VOLUME TELL: does relative volume in the first 30 min (quiet vs loud) separate
   recovery days from trapdoor days?

Also: what about a combined state (direction × volume)?
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
    df["mins"] = (df["ts"].dt.hour * 60 + df["ts"].dt.minute) - 570
    return df


def daystats(df):
    rows = []
    for dy, g in df.groupby("day"):
        if len(g) < 60:
            continue
        o = g["open"].iloc[0]; c = g["close"].iloc[-1]
        hi_i = g["high"].idxmax(); lo_i = g["low"].idxmin()

        first30 = g[g["mins"] < 30]
        if len(first30) < 4:
            continue
        f30_close = first30["close"].iloc[-1]
        f30_ret = f30_close / o - 1
        f30_low = first30["low"].min()
        f30_high = first30["high"].max()
        f30_vol = first30["volume"].sum()

        rest = g[g["mins"] >= 30]
        rest_vol = rest["volume"].sum() if len(rest) > 0 else 0
        day_vol = g["volume"].sum()

        first60 = g[g["mins"] < 60]
        f60_close = first60["close"].iloc[-1] if len(first60) >= 8 else np.nan
        f60_ret = f60_close / o - 1 if not np.isnan(f60_close) else np.nan

        rows.append(dict(
            day=dy, o=o, c=c, h=g["high"].max(), l=g["low"].min(),
            t_hi=g.loc[hi_i, "mins"], t_lo=g.loc[lo_i, "mins"],
            f30_ret=f30_ret, f30_vol=f30_vol, f30_low=f30_low, f30_high=f30_high,
            f60_ret=f60_ret,
            day_vol=day_vol,
        ))
    d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    d["pc"] = d["c"].shift(1)
    d["prev_dn"] = d["c"].shift(1) < d["c"].shift(2)
    d["gap"] = d["o"] / d["pc"] - 1
    d["oc"] = d["c"] / d["o"] - 1
    d["green"] = d["c"] > d["pc"]
    d["touch_pc"] = d["h"] >= d["pc"]
    d["avg_vol20"] = d["day_vol"].rolling(20).mean()
    d["f30_rvol"] = d["f30_vol"] / (d["avg_vol20"].shift(1) * (30/390))
    return d.dropna(subset=["pc", "gap", "f30_rvol"])


for sym in ["SPY", "QQQ"]:
    d = daystats(load(sym))
    DD = d[(d.prev_dn) & (d.gap < 0)].copy()
    print(f"\n{'='*60}")
    print(f" {sym} — prev-DOWN + gap-DOWN  (n={len(DD)})")
    print(f"{'='*60}")
    print(f" Baseline: P(O→C up)={DD.oc.gt(0).mean():.0%}  avg O→C={DD.oc.mean()*100:+.3f}%\n")

    # === TEST 1: first-30-min direction ===
    print(" TEST 1: First-30-min direction")
    print(f" {'state':<28s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}{'P(green)':>9s}{'medLOW':>8s}{'medHI':>8s}")
    for lbl, mask in [
        ("flush (f30 < -0.1%)",  DD.f30_ret < -0.001),
        ("flush (f30 < -0.2%)",  DD.f30_ret < -0.002),
        ("flush (f30 < -0.3%)",  DD.f30_ret < -0.003),
        ("mild down (-0.1..0%)", (DD.f30_ret >= -0.001) & (DD.f30_ret < 0)),
        ("bounce (f30 > 0%)",    DD.f30_ret >= 0),
        ("bounce (f30 > +0.1%)", DD.f30_ret >= 0.001),
        ("bounce (f30 > +0.2%)", DD.f30_ret >= 0.002),
    ]:
        S = DD[mask]
        if len(S) < 15:
            continue
        print(f" {lbl:<28s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%{S.green.mean():>9.0%}"
              f"{S.t_lo.median():>7.0f}m{S.t_hi.median():>7.0f}m")

    # === TEST 2: first-30-min volume (quiet vs loud) ===
    print(f"\n TEST 2: First-30-min relative volume (vs 20d avg proportional)")
    med_rvol = DD.f30_rvol.median()
    print(f" median f30 rVol = {med_rvol:.2f}")
    print(f" {'state':<28s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}{'P(green)':>9s}")
    for lbl, mask in [
        ("quiet (rVol < 0.8)",   DD.f30_rvol < 0.8),
        ("quiet (rVol < 1.0)",   DD.f30_rvol < 1.0),
        ("normal (0.8-1.5)",     (DD.f30_rvol >= 0.8) & (DD.f30_rvol < 1.5)),
        ("loud (rVol > 1.5)",    DD.f30_rvol >= 1.5),
        ("loud (rVol > 2.0)",    DD.f30_rvol >= 2.0),
        ("loud (rVol > 3.0)",    DD.f30_rvol >= 3.0),
    ]:
        S = DD[mask]
        if len(S) < 15:
            continue
        print(f" {lbl:<28s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%{S.green.mean():>9.0%}")

    # === TEST 3: combined (direction × volume) ===
    print(f"\n TEST 3: Combined first-30-min direction × volume")
    print(f" {'state':<40s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}{'P(green)':>9s}")
    combos = [
        ("flush + quiet",  (DD.f30_ret < -0.001) & (DD.f30_rvol < 1.0)),
        ("flush + loud",   (DD.f30_ret < -0.001) & (DD.f30_rvol >= 1.5)),
        ("bounce + quiet", (DD.f30_ret >= 0) & (DD.f30_rvol < 1.0)),
        ("bounce + loud",  (DD.f30_ret >= 0) & (DD.f30_rvol >= 1.5)),
        ("deep flush + quiet", (DD.f30_ret < -0.002) & (DD.f30_rvol < 1.0)),
        ("deep flush + loud",  (DD.f30_ret < -0.002) & (DD.f30_rvol >= 1.5)),
    ]
    for lbl, mask in combos:
        S = DD[mask]
        if len(S) < 15:
            continue
        print(f" {lbl:<40s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%{S.green.mean():>9.0%}")

    # === TEST 4: first-hour direction (60 min) for completeness ===
    print(f"\n TEST 4: First-hour direction (60 min)")
    DD60 = DD.dropna(subset=["f60_ret"])
    print(f" {'state':<28s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}")
    for lbl, mask in [
        ("f60 < -0.3%",  DD60.f60_ret < -0.003),
        ("f60 < -0.2%",  DD60.f60_ret < -0.002),
        ("f60 -0.2..0%", (DD60.f60_ret >= -0.002) & (DD60.f60_ret < 0)),
        ("f60 > 0%",     DD60.f60_ret >= 0),
        ("f60 > +0.2%",  DD60.f60_ret >= 0.002),
        ("f60 > +0.3%",  DD60.f60_ret >= 0.003),
    ]:
        S = DD60[mask]
        if len(S) < 15:
            continue
        print(f" {lbl:<28s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%")

    # === TEST 5: gap magnitude interaction with first-30-min ===
    print(f"\n TEST 5: Gap size × first-30-min direction")
    DD["gapb"] = pd.cut(DD.gap * 100, [-99, -0.5, -0.2, 0], labels=["<-0.5%", "-0.5..-0.2%", "-0.2..0%"])
    DD["f30d"] = np.where(DD.f30_ret < -0.001, "flush", np.where(DD.f30_ret >= 0, "bounce", "flat"))
    print(f" {'gap':<14s}{'f30':<10s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}")
    for gb in ["<-0.5%", "-0.5..-0.2%", "-0.2..0%"]:
        for fd in ["flush", "flat", "bounce"]:
            S = DD[(DD.gapb == gb) & (DD.f30d == fd)]
            if len(S) < 10:
                continue
            print(f" {gb:<14s}{fd:<10s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
                  f"{S.oc.mean()*100:>+9.3f}%")

    # === HALVES STABILITY for best cells ===
    print(f"\n HALVES STABILITY (2016-20 / 2021-26):")
    DD["half"] = [1 if str(x) < "2021-01-01" else 2 for x in DD["day"]]
    print(f" {'state':<40s}{'half':>5s}{'n':>5s}{'P(O→C up)':>11s}{'avg':>10s}")
    checks = [
        ("all down-down", DD.index == DD.index),
        ("flush f30<-0.1%", DD.f30_ret < -0.001),
        ("bounce f30>0%", DD.f30_ret >= 0),
        ("flush+quiet", (DD.f30_ret < -0.001) & (DD.f30_rvol < 1.0)),
        ("flush+loud", (DD.f30_ret < -0.001) & (DD.f30_rvol >= 1.5)),
    ]
    for lbl, mask in checks:
        for hf in [1, 2]:
            S = DD[mask & (DD.half == hf)]
            if len(S) < 10:
                continue
            print(f" {lbl:<40s}{hf:>5d}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
                  f"{S.oc.mean()*100:>+9.3f}%")
