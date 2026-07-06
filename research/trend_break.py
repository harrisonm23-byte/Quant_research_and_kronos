"""Which indicator marks a REAL intraday trend break?

For every band-walk (3+ closes below SMA9, %B<0.35, len>=5 bars), find the
first occurrence of each candidate break signal after the walk's 3rd bar,
then score:
  - P(no new session low for the rest of the day)   <- "trend actually broke"
  - P(up) and avg return +30m / +60m after signal
  - lateness: how far above the walk low the signal fires
Baseline for comparison: the naive close>SMA9 walk-end.
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


df = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
df = df.sort_values("ts").reset_index(drop=True)
d5 = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].reset_index(drop=True)
c = d5["close"]
d5["sma9"] = c.rolling(9).mean()
d5["mid20"] = c.rolling(20).mean()
sd = c.rolling(20).std()
d5["pctb"] = (c - (d5["mid20"] - 2 * sd)) / (4 * sd)
d5["rsi"] = wilder_rsi(c)
e12 = c.ewm(span=12, adjust=False).mean()
e26 = c.ewm(span=26, adjust=False).mean()
macd = e12 - e26
d5["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
d5["day"] = d5["ts"].dt.date
pv = d5["vwap"] * d5["volume"]
d5["svwap"] = pv.groupby(d5["day"]).cumsum() / d5["volume"].groupby(d5["day"]).cumsum()
d5["vol20"] = d5["volume"].rolling(20).mean()

o = d5["open"].values
h = d5["high"].values
l = d5["low"].values
cl = d5["close"].values
sma9 = d5["sma9"].values
mid20 = d5["mid20"].values
pctb = d5["pctb"].values
rsi = d5["rsi"].values
hist = d5["hist"].values
vw = d5["svwap"].values
volx = (d5["volume"] / d5["vol20"]).values
day = d5["day"].values
ts = d5["ts"].values
n = len(d5)
WARM = np.datetime64("2016-06-01")

day_end = {}
for i, dy in enumerate(day):
    day_end[dy] = i

below = (cl < sma9) & (pctb < 0.35)

walks = []
i = 2
while i < n - 1:
    if ts[i] < WARM or math.isnan(sma9[i]) or math.isnan(mid20[i]):
        i += 1
        continue
    if below[i] and below[i - 1] and below[i - 2] and day[i] == day[i - 2]:
        s = i - 2
        j = i
        de = day_end[day[s]]
        while j + 1 <= de and not (cl[j + 1] > sma9[j + 1]):
            j += 1
        if j - s + 1 >= 5:
            walks.append((s, min(j + 1, de), de))
        i = j + 2
    else:
        i += 1
print(f"{len(walks)} walks (len>=5 bars)")


def first_sig(name, s, e, de):
    """Return first index k in [s+3, de-1] where signal fires, else None."""
    lowest = l[s]
    low_i = s
    prev_low_rsi = None
    prev_low_px = None
    for k in range(s + 1, de):
        if l[k] < lowest:
            # bullish divergence check at new low
            if name == "E_rsi_diverge" and prev_low_rsi is not None:
                if rsi[k] > prev_low_rsi + 3 and l[k] < prev_low_px:
                    return k
            prev_low_rsi = rsi[k] if prev_low_rsi is None else min(prev_low_rsi, rsi[k]) \
                if False else rsi[k]
            prev_low_px = l[k]
            lowest = l[k]
            low_i = k
        if k < s + 3:
            continue
        if name == "A_close>sma9" and cl[k] > sma9[k]:
            return k
        if name == "B_close>mid20" and cl[k] > mid20[k]:
            return k
        if name == "C_close>vwap" and cl[k] > vw[k]:
            return k
        if name == "D_rsi30_recross" and rsi[k] > 30 and rsi[k - 1] <= 30:
            return k
        if name == "F_hist_rising3" and hist[k] < 0 and hist[k] > hist[k - 1] > hist[k - 2]:
            return k
        if (name == "G_climax_hammer" and volx[k] >= 3 and l[k] <= lowest
                and (h[k] - l[k]) > 0 and (cl[k] - l[k]) / (h[k] - l[k]) >= 0.6):
            return k
        if (name == "H_HH_HL" and h[k] > h[k - 1] and l[k] > l[k - 1]
                and cl[k] > o[k]):
            return k
    return None


SIGNALS = ["A_close>sma9", "B_close>mid20", "C_close>vwap", "D_rsi30_recross",
           "E_rsi_diverge", "F_hist_rising3", "G_climax_hammer", "H_HH_HL"]

print(f"{'signal':<18s} {'n':>5s} {'fire%':>6s} {'noNewLow':>9s} {'P+30m':>6s} "
      f"{'P+60m':>6s} {'avg60m':>8s} {'off-low':>8s}")
for name in SIGNALS:
    stats = []
    for s, e, de in walks:
        k = first_sig(name, s, e, de)
        if k is None or k + 6 >= n:
            continue
        walk_low = l[s:k + 1].min()
        rest_low = l[k + 1:de + 1].min() if k + 1 <= de else np.inf
        no_new_low = rest_low >= walk_low
        r30 = cl[k + 6] / cl[k] - 1 if day[k + 6] == day[k] else np.nan
        r60 = cl[k + 12] / cl[k] - 1 if k + 12 < n and day[k + 12] == day[k] else np.nan
        off_low = cl[k] / walk_low - 1
        stats.append((no_new_low, r30, r60, off_low))
    if not stats:
        print(f"{name:<18s}  none fired")
        continue
    nn_ = np.array([x[0] for x in stats])
    r30 = np.array([x[1] for x in stats])
    r60 = np.array([x[2] for x in stats])
    ol = np.array([x[3] for x in stats])
    r30v = r30[~np.isnan(r30)]
    r60v = r60[~np.isnan(r60)]
    print(f"{name:<18s} {len(stats):>5d} {len(stats)/len(walks):>6.0%} {nn_.mean():>9.1%} "
          f"{(r30v>0).mean():>6.1%} {(r60v>0).mean():>6.1%} {r60v.mean():>8.3%} {ol.mean():>8.3%}")
print("\nnoNewLow = P(session low is never broken again after the signal)")
print("off-low  = how far above the walk low the signal fires (lateness cost)")
