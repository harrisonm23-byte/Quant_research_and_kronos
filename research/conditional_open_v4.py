"""Conditional probabilities v4: broader state exploration.

Tests:
1. Consecutive down days (2, 3, 4+) → P(recovery), avg return
2. Day-of-week × gap direction
3. Gap-fill mechanics: P(gap fills by time bucket), conditional on gap size
4. Prior day's range (wide vs narrow) × gap direction
5. Open→first-hour high/low as stop/target for a structured trade
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
        hi = g["high"].max(); lo = g["low"].min()
        hi_i = g["high"].idxmax(); lo_i = g["low"].idxmin()

        first30 = g[g["mins"] < 30]
        first60 = g[g["mins"] < 60]
        first120 = g[g["mins"] < 120]
        last60 = g[g["mins"] >= 330]

        f30_hi = first30["high"].max() if len(first30) > 0 else np.nan
        f30_lo = first30["low"].min() if len(first30) > 0 else np.nan
        f60_hi = first60["high"].max() if len(first60) > 0 else np.nan
        f60_lo = first60["low"].min() if len(first60) > 0 else np.nan
        f120_hi = first120["high"].max() if len(first120) > 0 else np.nan
        f120_lo = first120["low"].min() if len(first120) > 0 else np.nan
        last60_hi = last60["high"].max() if len(last60) > 0 else np.nan
        last60_lo = last60["low"].min() if len(last60) > 0 else np.nan

        day_vol = g["volume"].sum()
        f30_vol = first30["volume"].sum() if len(first30) > 0 else 0

        rows.append(dict(
            day=dy, o=o, c=c, h=hi, l=lo, day_range=(hi-lo)/o,
            t_hi=g.loc[hi_i, "mins"], t_lo=g.loc[lo_i, "mins"],
            f30_hi=f30_hi, f30_lo=f30_lo, f60_hi=f60_hi, f60_lo=f60_lo,
            f120_hi=f120_hi, last60_hi=last60_hi, last60_lo=last60_lo,
            f120_lo=f120_lo,
            day_vol=day_vol, f30_vol=f30_vol,
            dow=pd.Timestamp(dy).dayofweek,
        ))
    d = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    d["pc"] = d["c"].shift(1)
    d["prev_dn"] = d["c"].shift(1) < d["c"].shift(2)
    d["prev2_dn"] = d["c"].shift(2) < d["c"].shift(3)
    d["prev3_dn"] = d["c"].shift(3) < d["c"].shift(4)
    d["gap"] = d["o"] / d["pc"] - 1
    d["oc"] = d["c"] / d["o"] - 1
    d["green"] = d["c"] > d["pc"]
    d["prev_range"] = d["day_range"].shift(1)
    d["prev_range_med"] = d["prev_range"].rolling(60).median()
    d["prev_range_wide"] = d["prev_range"] > d["prev_range_med"]
    d["gap_fill"] = np.where(d.gap < 0, d.h >= d.pc, d.l <= d.pc)
    d["gap_fill_f30"] = np.where(d.gap < 0, d.f30_hi >= d.pc, d.f30_lo <= d.pc)
    d["gap_fill_f60"] = np.where(d.gap < 0, d.f60_hi >= d.pc, d.f60_lo <= d.pc)
    d["gap_fill_f120"] = np.where(d.gap < 0, d.f120_hi >= d.pc,
                                  np.where(d.f120_lo.notna(), d.f120_lo <= d.pc, False))
    return d.dropna(subset=["pc", "gap", "prev2_dn"])


for sym in ["SPY", "QQQ"]:
    d = daystats(load(sym))
    print(f"\n{'='*70}")
    print(f" {sym} (n={len(d)} days)")
    print(f"{'='*70}")

    # === 1. Consecutive down days ===
    print(f"\n 1. CONSECUTIVE DOWN DAYS → next day")
    print(f" {'state':<30s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}{'P(green)':>9s}{'avg gap':>9s}")
    for lbl, mask in [
        ("1 down day",                (d.prev_dn) & (~d.prev2_dn)),
        ("2 consecutive down",        (d.prev_dn) & (d.prev2_dn) & (~d.prev3_dn)),
        ("3+ consecutive down",       (d.prev_dn) & (d.prev2_dn) & (d.prev3_dn)),
        ("1 down + gap down",         (d.prev_dn) & (~d.prev2_dn) & (d.gap < 0)),
        ("2 down + gap down",         (d.prev_dn) & (d.prev2_dn) & (~d.prev3_dn) & (d.gap < 0)),
        ("3+ down + gap down",        (d.prev_dn) & (d.prev2_dn) & (d.prev3_dn) & (d.gap < 0)),
        ("1 down + gap up",           (d.prev_dn) & (~d.prev2_dn) & (d.gap >= 0)),
        ("2 down + gap up",           (d.prev_dn) & (d.prev2_dn) & (~d.prev3_dn) & (d.gap >= 0)),
        ("3+ down + gap up",          (d.prev_dn) & (d.prev2_dn) & (d.prev3_dn) & (d.gap >= 0)),
    ]:
        S = d[mask]
        if len(S) < 20:
            continue
        print(f" {lbl:<30s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%{S.green.mean():>9.0%}"
              f"{S.gap.mean()*100:>+8.3f}%")

    # === 2. Day of week × direction ===
    print(f"\n 2. DAY OF WEEK (down-down days only)")
    DD = d[(d.prev_dn) & (d.gap < 0)]
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    print(f" {'day':<6s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}")
    for dw in range(5):
        S = DD[DD.dow == dw]
        if len(S) < 15:
            continue
        print(f" {dow_names[dw]:<6s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%")

    # === 3. Gap-fill timing ===
    print(f"\n 3. GAP-FILL TIMING (gap-down days)")
    GD = d[d.gap < 0].copy()
    GD["gapb"] = pd.cut(GD.gap * 100, [-99, -0.5, -0.2, -0.05, 0],
                         labels=["<-0.5%", "-0.5..-0.2%", "-0.2..-0.05%", "tiny"])
    print(f" {'gap size':<16s}{'n':>5s}{'fill 30m':>9s}{'fill 1h':>8s}{'fill 2h':>8s}{'fill day':>9s}{'avg O→C':>10s}")
    for gb, S in GD.groupby("gapb", observed=True):
        if len(S) < 30:
            continue
        print(f" {gb:<16s}{len(S):>5d}{S.gap_fill_f30.mean():>9.0%}{S.gap_fill_f60.mean():>8.0%}"
              f"{S.gap_fill_f120.mean():>8.0%}{S.gap_fill.mean():>9.0%}"
              f"{S.oc.mean()*100:>+9.3f}%")

    # Same for gap-up days
    print(f"\n    GAP-FILL TIMING (gap-up days)")
    GU = d[d.gap > 0].copy()
    GU["gapb"] = pd.cut(GU.gap * 100, [0, 0.05, 0.2, 0.5, 99],
                         labels=["tiny", "0.05..0.2%", "0.2..0.5%", ">0.5%"])
    print(f" {'gap size':<16s}{'n':>5s}{'fill 30m':>9s}{'fill 1h':>8s}{'fill 2h':>8s}{'fill day':>9s}{'avg O→C':>10s}")
    for gb, S in GU.groupby("gapb", observed=True):
        if len(S) < 30:
            continue
        print(f" {gb:<16s}{len(S):>5d}{S.gap_fill_f30.mean():>9.0%}{S.gap_fill_f60.mean():>8.0%}"
              f"{S.gap_fill_f120.mean():>8.0%}{S.gap_fill.mean():>9.0%}"
              f"{S.oc.mean()*100:>+9.3f}%")

    # === 4. Prior day's range × gap direction ===
    print(f"\n 4. PRIOR DAY RANGE (wide vs narrow) × gap direction")
    DR = d.dropna(subset=["prev_range_wide"])
    print(f" {'state':<30s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}")
    for lbl, mask in [
        ("narrow prev + gap down", (~DR.prev_range_wide) & (DR.gap < 0)),
        ("wide prev + gap down",   (DR.prev_range_wide) & (DR.gap < 0)),
        ("narrow prev + gap up",   (~DR.prev_range_wide) & (DR.gap >= 0)),
        ("wide prev + gap up",     (DR.prev_range_wide) & (DR.gap >= 0)),
        ("narrow + down + gap dn", (~DR.prev_range_wide) & (DR.prev_dn) & (DR.gap < 0)),
        ("wide + down + gap dn",   (DR.prev_range_wide) & (DR.prev_dn) & (DR.gap < 0)),
    ]:
        S = DR[mask]
        if len(S) < 20:
            continue
        print(f" {lbl:<30s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%")

    # === 5. Structured open trade: open→first-hour-low as stop, first-hour-high as target ===
    print(f"\n 5. STRUCTURED OPENING TRADE on down-down days")
    print(f"    Buy at open, stop = first-hour low, target = first-hour high")
    DD = d[(d.prev_dn) & (d.gap < 0)].copy()
    DD["stop_dist"] = (DD.f60_lo - DD.o) / DD.o
    DD["tgt_dist"] = (DD.f60_hi - DD.o) / DD.o
    DD["rr"] = DD.tgt_dist / DD.stop_dist.abs()
    DD_clean = DD[(DD.stop_dist < 0) & (DD.tgt_dist > 0)]  # normal days
    print(f" n={len(DD_clean)}  avg stop dist: {DD_clean.stop_dist.mean()*100:+.3f}%"
          f"  avg target dist: {DD_clean.tgt_dist.mean()*100:+.3f}%"
          f"  avg R:R = {DD_clean.rr.mean():.2f}")
    print(f" P(close > open): {DD_clean.oc.gt(0).mean():.0%}"
          f"  P(close > midpoint): {(DD_clean.c > (DD_clean.o + DD_clean.f60_lo)/2).mean():.0%}")

    # what if we use yesterday's close as target on gap-down?
    DD["pc_dist"] = (DD.pc - DD.o) / DD.o
    DD_gf = DD[DD.pc_dist > 0]  # gap is down so prev close is above open
    print(f"\n    Alternative: buy open, target = prev close (gap fill)")
    print(f" n={len(DD_gf)}  avg target (prevC-open)/open: {DD_gf.pc_dist.mean()*100:+.3f}%"
          f"  P(fills): {DD_gf.gap_fill.mean():.0%}"
          f"  avg O→C: {DD_gf.oc.mean()*100:+.3f}%")

    # === 6. 2-day pattern: down close + gap down + bounce = trade? ===
    print(f"\n 6. TWO-DAY COMBO: today = down-down + bounce → tomorrow")
    # Build: flag days where prev day was down-down AND bounced (oc > 0)
    d["prev_dd_bounce"] = (d.prev_dn) & (d.gap.shift(0) < 0) # this doesn't work, need prev day's OC
    # Actually need: was yesterday a down-down day that closed up?
    d["prev_oc"] = d["oc"].shift(1)
    d["prev_gap"] = d["gap"].shift(1)
    d["prev_dd"] = (d["prev_dn"].shift(1) == True) & (d["prev_gap"] < 0) if "prev_gap" in d.columns else False
    combo = d[(d["prev_oc"].shift(0) > 0) & (d.prev_dn) & (d.gap.shift(0) != d.gap.shift(0))]  # placeholder
    # Simpler: yesterday was down-down and recovered (O→C up), today's behavior
    yest_dd_bounce = (d.prev_dn.shift(1) == True) & (d.gap.shift(1) < 0) & (d.oc.shift(1) > 0)
    yest_dd_fail = (d.prev_dn.shift(1) == True) & (d.gap.shift(1) < 0) & (d.oc.shift(1) <= 0)
    print(f" {'yesterday state':<30s}{'n':>5s}{'P(O→C up)':>11s}{'avg O→C':>10s}{'P(green)':>9s}")
    for lbl, mask in [
        ("dd bounce → today",  yest_dd_bounce),
        ("dd fail → today",    yest_dd_fail),
    ]:
        S = d[mask].dropna(subset=["oc"])
        if len(S) < 15:
            continue
        print(f" {lbl:<30s}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
              f"{S.oc.mean()*100:>+9.3f}%{S.green.mean():>9.0%}")

    # === HALVES for notable cells ===
    print(f"\n HALVES STABILITY:")
    d["half"] = [1 if str(x) < "2021-01-01" else 2 for x in d["day"]]
    print(f" {'state':<30s}{'half':>5s}{'n':>5s}{'P(O→C up)':>11s}{'avg':>10s}")
    DD = d[(d.prev_dn) & (d.gap < 0)]
    checks = [
        ("2+ down + gap down", (d.prev_dn) & (d.prev2_dn) & (d.gap < 0)),
        ("3+ down + gap down", (d.prev_dn) & (d.prev2_dn) & (d.prev3_dn) & (d.gap < 0)),
    ]
    for lbl, mask in checks:
        for hf in [1, 2]:
            S = d[mask & (d.half == hf)]
            if len(S) < 10:
                continue
            print(f" {lbl:<30s}{hf:>5d}{len(S):>5d}{S.oc.gt(0).mean():>11.0%}"
                  f"{S.oc.mean()*100:>+9.3f}%")
