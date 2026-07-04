"""Entry-timing test: does intraday confirmation improve the daily QQQ longs?

For each historical trade of S4_IBS_QQQ_e20_x70 and S9_5DayLow_A_QQQ, keep the
exit fill fixed (from the original backtest) and re-price the ENTRY:

  base : buy 9:30 open (the original backtest's assumption)
  V1   : first 5m close > session VWAP (from 9:35 on) -> buy next bar open;
         fallback: buy 12:00 bar open if never confirmed by noon
  V1s  : same, but skip the trade entirely if not confirmed by noon
  V2   : just wait -- buy 10:00 bar open unconditionally
  V3   : dip limit at open-0.25% filled if touched before 10:30, else buy
         10:30 bar open

Slippage 0.02% on market fills; limit fills at limit price.
"""
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.0002

df = pd.read_csv(os.path.join(OUT, "QQQ_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
df = df.sort_values("ts").reset_index(drop=True)
d5 = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].reset_index(drop=True)
d5["day"] = d5["ts"].dt.date
pv = d5["vwap"] * d5["volume"]
d5["svwap"] = pv.groupby(d5["day"]).cumsum() / d5["volume"].groupby(d5["day"]).cumsum()

o = d5["open"].values
h = d5["high"].values
l = d5["low"].values
c = d5["close"].values
vw = d5["svwap"].values
tod = (d5["ts"].dt.hour * 60 + d5["ts"].dt.minute).values

day_rows = {}
for i, dy in enumerate(d5["day"].values):
    day_rows.setdefault(dy, []).append(i)


def entries_for_day(dy):
    rows = day_rows.get(dy)
    if not rows or len(rows) < 40:
        return None
    i0 = rows[0]
    out = {"base": o[i0] * (1 + SLIP)}
    # V1 / V1s
    conf = None
    for k in rows[1:]:
        if tod[k] >= 12 * 60:
            break
        if c[k] > vw[k] and k + 1 <= rows[-1]:
            conf = o[k + 1] * (1 + SLIP)
            break
    fb = next((k for k in rows if tod[k] >= 12 * 60), None)
    out["V1"] = conf if conf is not None else (o[fb] * (1 + SLIP) if fb else None)
    out["V1s"] = conf
    # V2: 10:00 open
    k10 = next((k for k in rows if tod[k] >= 10 * 60), None)
    out["V2"] = o[k10] * (1 + SLIP) if k10 else None
    # V3: dip limit -0.25% before 10:30, else 10:30 open
    lim = o[i0] * (1 - 0.0025)
    filled = None
    for k in rows:
        if tod[k] >= 10 * 60 + 30:
            break
        if l[k] <= lim:
            filled = lim
            break
    k1030 = next((k for k in rows if tod[k] >= 10 * 60 + 30), None)
    out["V3"] = filled if filled is not None else (o[k1030] * (1 + SLIP) if k1030 else None)
    return out


for csv, label in [("trades_S4_IBS_QQQ_e20_x70.csv", "IBS<0.20 QQQ"),
                   ("trades_S9_5DayLow_A_QQQ.csv", "5DayLow+IBS QQQ")]:
    tr = pd.read_csv(os.path.join(OUT, csv), parse_dates=["entry_date", "exit_date"])
    variants = {v: [] for v in ["base", "V1", "V1s", "V2", "V3"]}
    counts = {"n": 0, "V1_confirmed": 0, "V1_fallback": 0, "V1s_skipped": 0, "V3_limit": 0}
    for t in tr.itertuples():
        e = entries_for_day(t.entry_date.date())
        if e is None:
            continue
        counts["n"] += 1
        if e["V1s"] is not None:
            counts["V1_confirmed"] += 1
        else:
            counts["V1_fallback"] += 1
            counts["V1s_skipped"] += 1
        base_i0 = e["base"] / (1 + SLIP)
        if e["V3"] == base_i0 * (1 - 0.0025):
            counts["V3_limit"] += 1
        for v in variants:
            if e[v] is None:
                continue
            variants[v].append((t.exit_px / e[v] - 1, e[v]))
    print(f"\n===== {label}: {counts['n']} trades with 5m data =====")
    print(f"V1 confirmed by noon: {counts['V1_confirmed']} "
          f"({counts['V1_confirmed']/counts['n']:.0%}), fallback: {counts['V1_fallback']}; "
          f"V3 limit filled: {counts['V3_limit']} ({counts['V3_limit']/counts['n']:.0%})")
    base_avg = np.mean([r for r, _ in variants["base"]])
    print(f"{'variant':<6s} {'n':>4s} {'WR':>6s} {'avg ret':>8s} {'vs base':>8s} {'total':>8s}")
    for v, rows in variants.items():
        if not rows:
            continue
        a = np.array([r for r, _ in rows])
        total = np.prod(1 + a) - 1
        print(f"{v:<6s} {len(a):>4d} {(a>0).mean():>6.1%} {a.mean():>8.3%} "
              f"{(a.mean()-base_avg)*1e4:>+7.1f}bp {total:>8.1%}")
