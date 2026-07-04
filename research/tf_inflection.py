"""Test today's live observation: on a down >=0.5% day, when the 15m AND 30m
MACD histograms inflect upward simultaneously in the afternoon (both rising
vs their prior bar, from below zero), does price drift up into the close?

Controls: (a) down-days where alignment never fires, (b) 15m-only inflection.
SPY 5m, 10 years.
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
d5["day"] = d5["ts"].dt.date


def add_hist(frame):
    e12 = frame["close"].ewm(span=12, adjust=False).mean()
    e26 = frame["close"].ewm(span=26, adjust=False).mean()
    m = e12 - e26
    frame["hist"] = m - m.ewm(span=9, adjust=False).mean()
    return frame


def agg(step):
    minutes = (d5["ts"].dt.hour * 60 + d5["ts"].dt.minute) - 570
    grp = np.minimum(minutes // step, (390 // step) - 1)
    key = d5["day"].astype(str) + "_" + grp.astype(str)
    g = d5.groupby(key, sort=False)
    a = pd.DataFrame({"ts": g["ts"].first(), "close": g["close"].last(),
                      "end": g["ts"].last() + pd.Timedelta(minutes=5)})
    a = a.reset_index(drop=True).sort_values("ts").reset_index(drop=True)
    return add_hist(a)


a15 = agg(15)
a30 = agg(30)
ends15 = a15["end"].values
ends30 = a30["end"].values
h15 = a15["hist"].values
h30 = a30["hist"].values

cl = d5["close"].values
ts = d5["ts"].values
day = d5["day"].values
tod = (d5["ts"].dt.hour * 60 + d5["ts"].dt.minute).values
n = len(d5)
WARM = np.datetime64("2016-06-01")

# daily prev close map
days = sorted(set(day))
day_close = {}
for i, dy in enumerate(day):
    day_close[dy] = cl[i]          # overwritten -> last close of day
prev_close = {}
for k in range(1, len(days)):
    prev_close[days[k]] = day_close[days[k - 1]]

# day index ranges
day_rows = {}
for i, dy in enumerate(day):
    day_rows.setdefault(dy, []).append(i)


def hist_state(t, ends, H):
    k = np.searchsorted(ends, t, side="right") - 1
    if k < 1:
        return None, None
    return H[k], H[k - 1]


aligned, no_align, only15 = [], [], []
for dy in days:
    if dy not in prev_close:
        continue
    rows = day_rows[dy]
    if np.datetime64(pd.Timestamp(dy)) < WARM or len(rows) < 70:
        continue
    pc = prev_close[dy]
    # down >=0.5% at 13:30
    i1330 = next((i for i in rows if tod[i] >= 13 * 60 + 30), None)
    if i1330 is None or cl[i1330] / pc - 1 > -0.005:
        continue
    fired = None
    fired15 = None
    for i in rows:
        if tod[i] < 13 * 60 + 30 or tod[i] > 15 * 60:
            continue
        t = ts[i] + np.timedelta64(5, "m")
        c15, p15 = hist_state(t, ends15, h15)
        c30, p30 = hist_state(t, ends30, h30)
        if c15 is None or c30 is None:
            continue
        inf15 = c15 > p15 and p15 < 0
        inf30 = c30 > p30 and p30 < 0
        if inf15 and inf30 and fired is None:
            fired = i
        if inf15 and not inf30 and fired15 is None:
            fired15 = i
        if fired is not None:
            break
    eod = rows[-1]
    if fired is not None:
        aligned.append(cl[eod] / cl[fired] - 1)
    else:
        base_i = i1330
        no_align.append(cl[eod] / cl[base_i] - 1)
        if fired15 is not None:
            only15.append(cl[eod] / cl[fired15] - 1)

for name, arr in [("ALIGNED 15m+30m inflection", aligned),
                  ("15m-only inflection (no 30m)", only15),
                  ("down day, never aligned (from 13:30)", no_align)]:
    b = np.array(arr)
    if len(b):
        print(f"{name:<38s} n={len(b):>4d}  P(up into close)={ (b>0).mean():>6.1%}  "
              f"avg={b.mean():>+8.3%}  med={np.median(b):>+8.3%}")
