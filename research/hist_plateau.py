"""Does the MACD-histogram plateau ('large bars then flatness') carry info?

Signal: >=4 consecutive rising positive histogram bars, then the first
non-rising bar (the plateau tick). Mirror for negative/falling (red) runs.
Measured on SPY 5m RTH, 10y: forward 15/30/60min returns, P(up), and
whether a local price top forms within 30min (fade content).
Context split: did the run start from a bounce off a 2h low (like today)?
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")

df = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
df = df.sort_values("ts").reset_index(drop=True)
d5 = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].reset_index(drop=True)
c = d5["close"]
e12 = c.ewm(span=12, adjust=False).mean()
e26 = c.ewm(span=26, adjust=False).mean()
macd = e12 - e26
sig = macd.ewm(span=9, adjust=False).mean()
d5["hist"] = macd - sig
d5["day"] = d5["ts"].dt.date

H = d5["hist"].values
cl = d5["close"].values
h = d5["high"].values
l = d5["low"].values
day = d5["day"].values
ts = d5["ts"].values
n = len(d5)
WARM = np.datetime64("2016-06-01")

results = {"green_plateau": [], "red_plateau": []}
for i in range(30, n - 13):
    if ts[i] < WARM or day[i + 12] != day[i] or day[i - 6] != day[i]:
        continue
    # green: 4 rising positive bars then first non-rising
    if (H[i] <= H[i - 1] and H[i - 1] > H[i - 2] > H[i - 3] > H[i - 4]
            and H[i - 1] > 0 and H[i - 4] > 0):
        bounce = l[i - 24:i - 4].min() >= l[i - 4:i].min() if i >= 24 else False
        r15 = cl[i + 3] / cl[i] - 1
        r30 = cl[i + 6] / cl[i] - 1
        r60 = cl[i + 12] / cl[i] - 1
        # fade content: does price dip -0.10% before making +0.10%?
        dip_first = None
        for k in range(i + 1, min(i + 13, n)):
            if day[k] != day[i]:
                break
            if l[k] <= cl[i] * 0.999:
                dip_first = True
                break
            if h[k] >= cl[i] * 1.001:
                dip_first = False
                break
        results["green_plateau"].append((r15, r30, r60, dip_first, bounce))
    # red mirror
    if (H[i] >= H[i - 1] and H[i - 1] < H[i - 2] < H[i - 3] < H[i - 4]
            and H[i - 1] < 0 and H[i - 4] < 0):
        r15 = cl[i + 3] / cl[i] - 1
        r30 = cl[i + 6] / cl[i] - 1
        r60 = cl[i + 12] / cl[i] - 1
        pop_first = None
        for k in range(i + 1, min(i + 13, n)):
            if day[k] != day[i]:
                break
            if h[k] >= cl[i] * 1.001:
                pop_first = True
                break
            if l[k] <= cl[i] * 0.999:
                pop_first = False
                break
        results["red_plateau"].append((r15, r30, r60, pop_first, False))

for name, rows in results.items():
    if not rows:
        continue
    r15 = np.array([x[0] for x in rows])
    r30 = np.array([x[1] for x in rows])
    r60 = np.array([x[2] for x in rows])
    first = [x[3] for x in rows if x[3] is not None]
    print(f"=== {name}: n={len(rows)} (~{len(rows)/2500:.1f}/day) ===")
    print(f"  +15m: P(up)={ (r15>0).mean():.1%}  avg={r15.mean():+.4%}")
    print(f"  +30m: P(up)={ (r30>0).mean():.1%}  avg={r30.mean():+.4%}")
    print(f"  +60m: P(up)={ (r60>0).mean():.1%}  avg={r60.mean():+.4%}")
    if name == "green_plateau":
        print(f"  P(-0.10% dip before +0.10% pop) within 60m: {np.mean(first):.1%}  (n={len(first)})")
        b = np.array([x[4] for x in rows])
        r30b, r30nb = r30[b], r30[~b]
        if b.sum() >= 30:
            print(f"  bounce-context (like today): n={b.sum()}  +30m avg={r30b.mean():+.4%}  "
                  f"P(up)={(r30b>0).mean():.1%}")
            print(f"  other context:               n={(~b).sum()}  +30m avg={r30nb.mean():+.4%}  "
                  f"P(up)={(r30nb>0).mean():.1%}")
    else:
        print(f"  P(+0.10% pop before -0.10% dip) within 60m: {np.mean(first):.1%}  (n={len(first)})")

# baseline
base_mask = (ts >= WARM)
idxs = np.arange(30, n - 13)[::7]
b30 = []
for i in idxs:
    if ts[i] >= WARM and day[i + 6] == day[i]:
        b30.append(cl[i + 6] / cl[i] - 1)
b30 = np.array(b30)
print(f"\nbaseline any-bar +30m: P(up)={(b30>0).mean():.1%}  avg={b30.mean():+.4%}  (n={len(b30)})")
