"""Band-walk study on SPY 5m (10y): the corridor between SMA9 and lower BB.

Walk definition: 3+ consecutive 5m closes below SMA9 with %B < 0.35.
Measured:
  A. persistence: bars until first close back above SMA9
  B. the 'SMA9 kiss' short: during an active walk, bar HIGH touches SMA9
     -> short next bar open; exit on first close above SMA9 (walk over) or EOD
  C. what follows walk END (close back above SMA9): next 30/60min drift
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.0002

df = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
df = df.sort_values("ts").reset_index(drop=True)
d5 = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].reset_index(drop=True)
c = d5["close"]
d5["sma9"] = c.rolling(9).mean()
mid = c.rolling(20).mean()
sd = c.rolling(20).std()
d5["pctb"] = (c - (mid - 2 * sd)) / (4 * sd)
d5["day"] = d5["ts"].dt.date

o = d5["open"].values
h = d5["high"].values
cl = d5["close"].values
sma9 = d5["sma9"].values
pctb = d5["pctb"].values
day = d5["day"].values
n = len(d5)
WARM = np.datetime64("2016-06-01")
ts = d5["ts"].values

below = (cl < sma9) & (pctb < 0.35)

# ---- A: walk persistence ----
walks = []          # (start_i, end_i)  end = first close > sma9 (or day end)
i = 1
while i < n - 1:
    if ts[i] < WARM or math.isnan(sma9[i]):
        i += 1
        continue
    if below[i] and below[i - 1] and i - 2 >= 0 and below[i - 2] and day[i] == day[i - 2]:
        s = i - 2
        j = i
        while j + 1 < n and day[j + 1] == day[s] and not (cl[j + 1] > sma9[j + 1]):
            j += 1
        walks.append((s, min(j + 1, n - 1)))
        i = j + 2
    else:
        i += 1

lengths = [e - s for s, e in walks]
rets = [cl[e] / cl[s] - 1 for s, e in walks]
print(f"A. WALKS: {len(walks)} over 10y ({len(walks)/2500:.1f}/day avg)")
print(f"   length until close>SMA9: median {np.median(lengths):.0f} bars, "
      f"mean {np.mean(lengths):.1f}, p90 {np.percentile(lengths, 90):.0f} "
      f"(= {np.percentile(lengths,90)*5:.0f} min)")
print(f"   price drift start->end: mean {np.mean(rets):+.3%}  "
      f"(negative = walk kept paying the short side)")

# ---- B: SMA9-kiss short during walk ----
kiss = []
for s, e in walks:
    for k in range(s + 1, e):
        if h[k] >= sma9[k] and cl[k] < sma9[k]:      # touched from below, rejected
            entry = o[k + 1] * (1 - SLIP)
            xi = e                                   # exit at walk end close
            ret = entry / (cl[xi] * (1 + SLIP)) - 1  # short return
            kiss.append(ret)
            break                                    # first kiss per walk only
a = np.array(kiss)
if len(a):
    wins = a[a > 0]
    losses = a[a <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    print(f"\nB. SMA9-KISS SHORT (first kiss per walk, exit at walk end): n={len(a)}")
    print(f"   WR={(a>0).mean():.1%}  avg={a.mean():+.3%}  med={np.median(a):+.3%}  PF={pf:.2f}")

# ---- C: after the walk ends ----
after30, after60 = [], []
for s, e in walks:
    if e + 6 < n and day[e + 6] == day[e]:
        after30.append(cl[e + 6] / cl[e] - 1)
    if e + 12 < n and day[e + 12] == day[e]:
        after60.append(cl[e + 12] / cl[e] - 1)
for lbl, arr in [("+30min", after30), ("+60min", after60)]:
    b = np.array(arr)
    print(f"\nC. after walk END {lbl}: n={len(b)}  P(up)={ (b>0).mean():.1%}  avg={b.mean():+.3%}"
          if lbl == "+30min" else
          f"   after walk END {lbl}: n={len(b)}  P(up)={(b>0).mean():.1%}  avg={b.mean():+.3%}")
