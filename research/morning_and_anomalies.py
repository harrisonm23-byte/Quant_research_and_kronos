"""B. Morning moves: gap & first-30m behavior -> rest of day (SPY, 10y).
C. Published anomalies on our data: overnight-vs-intraday drift, turn-of-month.
"""
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")

# ---------- B. morning moves ----------
d5 = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
d5["ts"] = pd.to_datetime(d5["timestamps"]).dt.tz_convert(NY)
d5 = d5[(d5["ts"].dt.time >= dtime(9, 30)) & (d5["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
d5["day"] = d5["ts"].dt.date

days = []
prev_close = None
for dy, g in d5.groupby("day", sort=True):
    g = g.reset_index(drop=True)
    if len(g) < 70:
        prev_close = g["close"].iloc[-1] if len(g) else prev_close
        continue
    o = g["open"].iloc[0]
    c30 = g["close"].iloc[5]        # 10:00 close
    eod = g["close"].iloc[-1]
    lo_am = g["low"].iloc[0:6].min()
    hi_am = g["high"].iloc[0:6].max()
    row = dict(day=dy, open=o, c30=c30, eod=eod,
               gap=(o / prev_close - 1) if prev_close else np.nan,
               r30=c30 / o - 1, rest=eod / c30 - 1,
               oc=eod / o - 1,
               gap_filled=(lo_am <= prev_close if prev_close and o > prev_close
                           else (hi_am >= prev_close if prev_close else np.nan)))
    days.append(row)
    prev_close = eod
D = pd.DataFrame(days).dropna(subset=["gap"])
D = D[D["day"] >= pd.Timestamp("2016-06-01").date()]
print(f"B. MORNING MOVES — SPY, {len(D)} days")

print("\n  gap size -> rest of day (open->close):")
D["gapb"] = pd.cut(D["gap"] * 100, [-9, -0.5, -0.1, 0.1, 0.5, 9],
                   labels=["gap<-0.5%", "-0.5..-0.1", "flat", "+0.1..0.5", "gap>+0.5%"])
for b, g in D.groupby("gapb", observed=True):
    print(f"    {b:<11s} n={len(g):>4d}  P(oc up)={(g['oc']>0).mean():>6.1%}  avg oc={g['oc'].mean():>+7.3%}"
          f"  P(gap filled same day)={g['gap_filled'].mean():>6.1%}")

print("\n  first 30 min -> rest of day (10:00 -> close), by gap context:")
D["r30b"] = np.where(D["r30"] > 0.002, "30m up>0.2%", np.where(D["r30"] < -0.002, "30m dn>0.2%", "30m flat"))
for (gb, rb), g in D.groupby(["gapb", "r30b"], observed=True):
    if len(g) < 80:
        continue
    print(f"    {str(gb):<11s} + {rb:<12s} n={len(g):>4d}  P(rest up)={(g['rest']>0).mean():>6.1%}  "
          f"avg rest={g['rest'].mean():>+7.3%}")

# ---------- C. anomalies ----------
print("\nC. PUBLISHED ANOMALIES ON OUR DATA")
for sym in ["SPY", "QQQ"]:
    dd = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"), parse_dates=["date"])
    dd = dd[dd["date"] >= "2016-06-01"].reset_index(drop=True)
    on = dd["open"] / dd["close"].shift(1) - 1      # overnight
    ic = dd["close"] / dd["open"] - 1               # intraday
    yrs = (dd["date"].iloc[-1] - dd["date"].iloc[0]).days / 365.25
    on_t = np.prod(1 + on.dropna()) ** (1 / yrs) - 1
    ic_t = np.prod(1 + ic.dropna()) ** (1 / yrs) - 1
    print(f"  {sym}: OVERNIGHT (close->open) ann = {on_t:+.1%}   INTRADAY (open->close) ann = {ic_t:+.1%}")

    dd["tom"] = False
    dates = dd["date"].dt.to_period("M")
    for m in dates.unique():
        idxs = dd.index[dates == m]
        if len(idxs) >= 5:
            dd.loc[idxs[-2]:, "tom"] = True   # last 2 of month
            dd.loc[idxs[:3], "tom"] = True    # first 3 of month
    dd.loc[dd.index[-2:], "tom"] = True
    r = dd["close"].pct_change()
    tom, rest = r[dd["tom"]], r[~dd["tom"]]
    print(f"       TURN-OF-MONTH (last2+first3): avg {tom.mean():+.4%}/day (n={len(tom)})  "
          f"vs rest {rest.mean():+.4%}/day (n={len(rest)})")
