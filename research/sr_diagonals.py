"""Phase C: diagonal trendlines on SPY 5m (same-day lines, 2016-2026).

Up-trendline: two most recent same-day confirmed pivot lows (K=6), second HIGHER,
>=60min apart, slope < 0.5*ATR/bar. Zone = projected line +/-0.05%.
Max 3 projected touches; dead on close 0.1% below line. Down-trendline mirrored.
Events: TEST (race +/-0.25%, 90min) and BREAK (continuation race). Volume split.
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
K = 6
TGT = 0.0025
HORIZON = 18
WZ = 0.0005     # +/-0.05%

df = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
df["day"] = df["ts"].dt.date
c = df["close"]
h_, l_ = df["high"], df["low"]
tr = np.maximum(h_ - l_, np.maximum((h_ - c.shift(1)).abs(), (l_ - c.shift(1)).abs()))
df["atr20"] = tr.rolling(20).mean()
df["vol20"] = df["volume"].rolling(20).mean()
hh = df["high"].values; ll = df["low"].values; cl = df["close"].values
atr = df["atr20"].values; volx = (df["volume"] / df["vol20"]).values
day = df["day"].values
ts_arr = df["ts"].values
n = len(df)
rows_of = {}
for i, dy in enumerate(day):
    rows_of.setdefault(dy, []).append(i)
WARM = np.datetime64("2016-06-01")


def race(i, direction):
    px = cl[i]
    up, dn = px * (1 + TGT), px * (1 - TGT)
    de = rows_of[day[i]][-1]
    for j in range(i + 1, min(i + HORIZON + 1, de + 1)):
        u, d_ = hh[j] >= up, ll[j] <= dn
        if u and d_:
            return 0
        if u:
            return direction
        if d_:
            return -direction
    return 0


events = []
for dy, rows in rows_of.items():
    if len(rows) < 60 or np.datetime64(pd.Timestamp(dy)) < WARM:
        continue
    r0 = rows[0]
    plows, phighs = [], []      # (bar, price)
    upline = dnline = None      # dict(a_bar, a_px, slope, touches, outside)
    for i in rows:
        j = i - K
        if j - K >= r0:
            if ll[j] == min(ll[j - K:j + K + 1]):
                plows.append((j, ll[j]))
                if len(plows) >= 2:
                    (b1, p1), (b2, p2) = plows[-2], plows[-1]
                    if p2 > p1 and b2 - b1 >= 12:
                        slope = (p2 - p1) / (b2 - b1)
                        if not math.isnan(atr[i]) and slope < 0.5 * atr[i]:
                            upline = dict(ab=b2, ap=p2, s=slope, t=0, out=99)
            if hh[j] == max(hh[j - K:j + K + 1]):
                phighs.append((j, hh[j]))
                if len(phighs) >= 2:
                    (b1, p1), (b2, p2) = phighs[-2], phighs[-1]
                    if p2 < p1 and b2 - b1 >= 12:
                        slope = (p1 - p2) / (b2 - b1)
                        if not math.isnan(atr[i]) and slope < 0.5 * atr[i]:
                            dnline = dict(ab=b2, ap=p2, s=-slope, t=0, out=99)
        px = cl[i]
        if upline is not None and i > upline["ab"]:
            v = upline["ap"] + upline["s"] * (i - upline["ab"])
            w = WZ * px
            if cl[i] < v - 0.001 * px:
                out = race(i, -1)
                if out != 0:
                    events.append(("upline_BREAK", out > 0, volx[i], ts_arr[i]))
                upline = None
            elif ll[i] <= v + w and upline["out"] >= 6:
                out = race(i, +1)
                if out != 0:
                    events.append(("upline_TEST", out > 0, volx[i], ts_arr[i]))
                upline["t"] += 1
                upline["out"] = 0
                if upline["t"] >= 3:
                    upline = None
            else:
                upline["out"] += 1
        if dnline is not None and i > dnline["ab"]:
            v = dnline["ap"] + dnline["s"] * (i - dnline["ab"])
            w = WZ * px
            if cl[i] > v + 0.001 * px:
                out = race(i, +1)
                if out != 0:
                    events.append(("dnline_BREAK", out > 0, volx[i], ts_arr[i]))
                dnline = None
            elif hh[i] >= v - w and dnline["out"] >= 6:
                out = race(i, -1)
                if out != 0:
                    events.append(("dnline_TEST", out > 0, volx[i], ts_arr[i]))
                dnline["t"] += 1
                dnline["out"] = 0
                if dnline["t"] >= 3:
                    dnline = None
            else:
                dnline["out"] += 1

E = pd.DataFrame(events, columns=["kind", "win", "volx", "ts"])
E["half"] = np.where(E["ts"] < np.datetime64("2022-01-01"), "H1", "H2")
print(f"diagonal events: {len(E)}")
print(f"(baselines from v2: up-first 45.0%, down-first 55.0%)\n")
print(f"{'event':<14s}{'n':>6s}{'P(favorable)':>13s}{'H1':>7s}{'H2':>7s}{'quiet<=1.2':>11s}{'loud>1.5':>10s}")
for kind, g in E.groupby("kind"):
    q = g[g.volx <= 1.2]["win"]
    ld = g[g.volx > 1.5]["win"]
    print(f"{kind:<14s}{len(g):>6d}{g['win'].mean():>13.1%}"
          f"{g[g.half=='H1']['win'].mean():>7.1%}{g[g.half=='H2']['win'].mean():>7.1%}"
          f"{q.mean() if len(q)>=30 else float('nan'):>11.1%}"
          f"{ld.mean() if len(ld)>=30 else float('nan'):>10.1%}")
