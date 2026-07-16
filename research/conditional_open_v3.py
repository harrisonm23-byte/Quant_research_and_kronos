"""Conditional open v3: tradeable return from the 30-min mark onward.

The v2 study found that first-30-min direction splits down-down days into
70% recovery (bounce) vs 33% (flush). But a trader can't capture the open →
close return — they observe the first 30 min and THEN buy. So what's the
return from 10:00 to close? That's the real edge.

Also: what about buying at the 30-min low (the flush low) on flush days?
Does a reversal trade work on the subset that does recover?
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


def daystats_extended(df):
    rows = []
    for dy, g in df.groupby("day"):
        if len(g) < 60:
            continue
        o = g["open"].iloc[0]; c = g["close"].iloc[-1]

        first30 = g[g["mins"] < 30]
        after30 = g[g["mins"] >= 30]
        if len(first30) < 4 or len(after30) < 40:
            continue

        f30_close = first30["close"].iloc[-1]
        f30_ret = f30_close / o - 1
        f30_low = first30["low"].min()
        f30_high = first30["high"].max()
        f30_vol = first30["volume"].sum()

        a30_close = after30["close"].iloc[-1]  # = close
        a30_ret = a30_close / f30_close - 1     # return from 10:00 to close

        first60 = g[g["mins"] < 60]
        after60 = g[g["mins"] >= 60]
        if len(first60) >= 8 and len(after60) >= 40:
            f60_close = first60["close"].iloc[-1]
            f60_ret = f60_close / o - 1
            a60_ret = after60["close"].iloc[-1] / f60_close - 1
        else:
            f60_ret = np.nan; a60_ret = np.nan

        hi_i = g["high"].idxmax(); lo_i = g["low"].idxmin()
        day_vol = g["volume"].sum()
        after30_hi = after30["high"].max()
        after30_lo = after30["low"].min()
        after30_mfe_long = after30_hi / f30_close - 1
        after30_mae_long = after30_lo / f30_close - 1

        rows.append(dict(
            day=dy, o=o, c=c, h=g["high"].max(), l=g["low"].min(),
            t_hi=g.loc[hi_i, "mins"], t_lo=g.loc[lo_i, "mins"],
            f30_ret=f30_ret, f30_close=f30_close, f30_vol=f30_vol,
            f60_ret=f60_ret,
            a30_ret=a30_ret, a60_ret=a60_ret,
            a30_mfe=after30_mfe_long, a30_mae=after30_mae_long,
            day_vol=day_vol,
        ))
    d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    d["pc"] = d["c"].shift(1)
    d["prev_dn"] = d["c"].shift(1) < d["c"].shift(2)
    d["gap"] = d["o"] / d["pc"] - 1
    d["oc"] = d["c"] / d["o"] - 1
    d["avg_vol20"] = d["day_vol"].rolling(20).mean()
    d["f30_rvol"] = d["f30_vol"] / (d["avg_vol20"].shift(1) * (30/390))
    return d.dropna(subset=["pc", "gap", "f30_rvol"])


for sym in ["SPY", "QQQ"]:
    d = daystats_extended(load(sym))
    DD = d[(d.prev_dn) & (d.gap < 0)].copy()
    print(f"\n{'='*70}")
    print(f" {sym} — TRADEABLE RETURNS from 10:00 AM → close")
    print(f"{'='*70}")

    print(f"\n TRADE: buy at 10:00 AM (after observing first 30 min)")
    print(f" {'state':<28s}{'n':>5s}{'P(+)':>6s}{'avg 10→C':>10s}{'med 10→C':>10s}{'avgMFE':>8s}{'avgMAE':>8s}")
    for lbl, mask in [
        ("ALL down-down",          DD.index == DD.index),
        ("flush (f30 < -0.1%)",    DD.f30_ret < -0.001),
        ("flush (f30 < -0.2%)",    DD.f30_ret < -0.002),
        ("flush (f30 < -0.3%)",    DD.f30_ret < -0.003),
        ("mild down (-0.1..0%)",   (DD.f30_ret >= -0.001) & (DD.f30_ret < 0)),
        ("bounce (f30 > 0%)",      DD.f30_ret >= 0),
        ("bounce (f30 > +0.1%)",   DD.f30_ret >= 0.001),
        ("bounce (f30 > +0.2%)",   DD.f30_ret >= 0.002),
    ]:
        S = DD[mask]
        if len(S) < 15:
            continue
        print(f" {lbl:<28s}{len(S):>5d}{S.a30_ret.gt(0).mean():>6.0%}"
              f"{S.a30_ret.mean()*100:>+9.3f}%{S.a30_ret.median()*100:>+9.3f}%"
              f"{S.a30_mfe.mean()*100:>+7.2f}%{S.a30_mae.mean()*100:>+7.2f}%")

    print(f"\n TRADE: buy at 10:30 AM (after observing first hour)")
    DD60 = DD.dropna(subset=["a60_ret"])
    print(f" {'state':<28s}{'n':>5s}{'P(+)':>6s}{'avg 10:30→C':>12s}")
    for lbl, mask in [
        ("ALL down-down",        DD60.index == DD60.index),
        ("f60 < -0.3%",         DD60.f60_ret < -0.003),
        ("f60 < -0.2%",         DD60.f60_ret < -0.002),
        ("f60 -0.2..0%",        (DD60.f60_ret >= -0.002) & (DD60.f60_ret < 0)),
        ("f60 > 0%",            DD60.f60_ret >= 0),
        ("f60 > +0.2%",         DD60.f60_ret >= 0.002),
        ("f60 > +0.3%",         DD60.f60_ret >= 0.003),
    ]:
        S = DD60[mask]
        if len(S) < 15:
            continue
        print(f" {lbl:<28s}{len(S):>5d}{S.a60_ret.gt(0).mean():>6.0%}"
              f"{S.a60_ret.mean()*100:>+11.3f}%")

    # Race analysis: +0.20% vs -0.20% within remaining session for the bounce state
    print(f"\n RACE: ±0.20% from 10:00 on bounce days (f30 > 0%)")
    BOUNCE = DD[DD.f30_ret >= 0]
    hit_up = (BOUNCE.a30_mfe >= 0.002).sum()
    hit_dn = (BOUNCE.a30_mae <= -0.002).sum()
    both = ((BOUNCE.a30_mfe >= 0.002) & (BOUNCE.a30_mae <= -0.002)).sum()
    n = len(BOUNCE)
    print(f" n={n}  hit +0.20%: {hit_up} ({hit_up/n:.0%})  hit -0.20%: {hit_dn} ({hit_dn/n:.0%})")
    print(f" hit both: {both}  (can't determine order from OHLC)")

    # Race at ±0.15%
    hit_up15 = (BOUNCE.a30_mfe >= 0.0015).sum()
    hit_dn15 = (BOUNCE.a30_mae <= -0.0015).sum()
    print(f" at ±0.15%: hit +: {hit_up15} ({hit_up15/n:.0%})  hit -: {hit_dn15} ({hit_dn15/n:.0%})")

    # Halves for the tradeable return
    print(f"\n HALVES: tradeable 10:00→close return")
    DD["half"] = [1 if str(x) < "2021-01-01" else 2 for x in DD["day"]]
    print(f" {'state':<28s}{'half':>5s}{'n':>5s}{'P(+)':>6s}{'avg':>10s}")
    for lbl, mask in [
        ("ALL down-down",      DD.index == DD.index),
        ("bounce f30>0%",      DD.f30_ret >= 0),
        ("flush f30<-0.1%",    DD.f30_ret < -0.001),
    ]:
        for hf in [1, 2]:
            S = DD[mask & (DD.half == hf)]
            if len(S) < 10:
                continue
            print(f" {lbl:<28s}{hf:>5d}{len(S):>5d}{S.a30_ret.gt(0).mean():>6.0%}"
                  f"{S.a30_ret.mean()*100:>+9.3f}%")

    # Year breakdown for bounce state
    print(f"\n YEAR: bounce (f30>0%) tradeable 10→close")
    DD["year"] = [str(x)[:4] for x in DD["day"]]
    BOUNCE = DD[DD.f30_ret >= 0]
    print(f" {'year':<6s}{'n':>5s}{'P(+)':>6s}{'avg':>10s}")
    for yr, S in BOUNCE.groupby("year"):
        if len(S) < 10:
            continue
        print(f" {yr:<6s}{len(S):>5d}{S.a30_ret.gt(0).mean():>6.0%}{S.a30_ret.mean()*100:>+9.3f}%")
