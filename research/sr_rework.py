"""Rework the loud-volume-resistance cell into something usable (5m).

Events: resistance TEST with volx>1.5 (same zone engine as v2), both symbols.
Entry variants:  V0 at touch close | V1 first 5m close above zone top (<=1h)
                 | V2 first 15m-boundary close above zone top (<=1.5h)
Outcome frames:  +0.25/-0.25 (18b) | +0.50/-0.25 (36b) | +0.50/-0.50 (36b) | EOD
Net of 4bp round trip. Halves split on the best cells.
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
K = 6; CLUSTER = 0.001; MINSEP = 12; OUTSIDE = 6
COST = 0.0004


def load(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
    df["day"] = df["ts"].dt.date
    c = df["close"]
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - c.shift(1)).abs(), (df["low"] - c.shift(1)).abs()))
    df["atr20"] = tr.rolling(20).mean()
    df["vol20"] = df["volume"].rolling(20).mean()
    return df


def resistance_events(df):
    hh = df["high"].values; ll = df["low"].values; cl = df["close"].values
    atr = df["atr20"].values; volx = (df["volume"] / df["vol20"]).values
    day = df["day"].values; ts_arr = df["ts"].values
    n = len(df)
    rows_of = {}
    for i, dy in enumerate(day):
        rows_of.setdefault(dy, []).append(i)
    zones = []
    WARM = np.datetime64("2016-06-01")

    class Z:
        __slots__ = ("center", "pivots", "valid", "outside", "side", "broken", "last")
        def __init__(self, ctr, i):
            self.center = ctr; self.pivots = [i]; self.valid = False
            self.outside = 99; self.side = None; self.broken = False; self.last = i

    events = []
    cur = None
    for i in range(30, n):
        if day[i] != cur:
            cur = day[i]
            zones = [z for z in zones if i - z.last < 5 * 78][-25:]
        if ts_arr[i] < WARM or math.isnan(atr[i]):
            continue
        j = i - K
        if j - K >= 0 and day[j - K] == day[i]:
            for val in (ll[j], hh[j]):
                is_ext = (val == min(ll[j - K:j + K + 1])) or (val == max(hh[j - K:j + K + 1]))
                if not is_ext:
                    continue
                hit = False
                for z in zones:
                    if abs(z.center - val) / val < CLUSTER:
                        if not z.valid and (j - z.pivots[-1]) >= MINSEP:
                            z.valid = True
                        z.pivots.append(j); z.last = i
                        z.center = 0.7 * z.center + 0.3 * val
                        hit = True
                        break
                if not hit:
                    zones.append(Z(val, j))
        px = cl[i]
        w = max(0.0005 * px, 0.20 * atr[i])
        for z in zones:
            if not z.valid or z.broken:
                continue
            top, bot = z.center + w, z.center - w
            in_z = ll[i] <= top and hh[i] >= bot
            if in_z and z.outside >= OUTSIDE and z.side == "below":
                if volx[i] > 1.5:
                    events.append((i, top))
                z.outside = 0
            elif not in_z:
                z.outside += 1
                if z.side == "below" and cl[i] > top + 0.0005 * px:
                    z.broken = True
                elif z.side == "above" and cl[i] < bot - 0.0005 * px:
                    z.broken = True
                z.side = "above" if cl[i] > top else ("below" if cl[i] < bot else z.side)
            else:
                z.outside = 0
    return events, rows_of


def frames(df, events, rows_of):
    hh = df["high"].values; ll = df["low"].values; cl = df["close"].values
    day = df["day"].values; ts = df["ts"]
    tod_end = (ts.dt.hour * 60 + ts.dt.minute + 5).values
    out = {}

    def entry_bar(i, top, mode):
        if mode == "V0":
            return i
        de = rows_of[day[i]][-1]
        lim = 12 if mode == "V1" else 18
        for j in range(i + 1, min(i + lim + 1, de + 1)):
            if cl[j] > top + 0.0005 * cl[j]:
                if mode == "V1":
                    return j
                if mode == "V2" and tod_end[j] % 15 == 0:
                    return j
        return None

    for mode in ["V0", "V1", "V2"]:
        for tgt, stp, hz, lbl in [(0.0025, 0.0025, 18, "+.25/-.25"),
                                  (0.005, 0.0025, 36, "+.50/-.25"),
                                  (0.005, 0.005, 36, "+.50/-.50"),
                                  (None, None, None, "EOD")]:
            rets = []
            n_missed = 0
            for (i, top) in events:
                eb = entry_bar(i, top, mode)
                if eb is None:
                    n_missed += 1
                    continue
                e = cl[eb]
                de = rows_of[day[eb]][-1]
                r = None
                if lbl == "EOD":
                    r = cl[de] / e - 1
                else:
                    up, dn = e * (1 + tgt), e * (1 - stp)
                    for j2 in range(eb + 1, min(eb + hz + 1, de + 1)):
                        if hh[j2] >= up and ll[j2] <= dn:
                            r = -stp; break
                        if hh[j2] >= up:
                            r = tgt; break
                        if ll[j2] <= dn:
                            r = -stp; break
                    if r is None:
                        r = cl[min(eb + hz, de)] / e - 1
                rets.append(r - COST)
            a = np.array(rets)
            key = (mode, lbl)
            out[key] = (len(a), n_missed, (a > 0).mean() if len(a) else np.nan,
                        a.mean() if len(a) else np.nan)
    return out


for sym in ["SPY", "QQQ"]:
    df = load(sym)
    ev, rows_of = resistance_events(df)
    print(f"\n===== {sym}: loud-volume resistance tests = {len(ev)} events =====")
    res = frames(df, ev, rows_of)
    print(f"{'entry':<5s}{'frame':<11s}{'n':>6s}{'missed':>7s}{'WR':>7s}{'avg net':>9s}")
    for (mode, lbl), (n_, m_, wr, avg) in res.items():
        print(f"{mode:<5s}{lbl:<11s}{n_:>6d}{m_:>7d}{wr:>7.1%}{avg*100:>8.3f}%")
    # halves for the headline frames on V0
    ts_ev = df["ts"].values
    for lbl_want in ["+.50/-.25", "EOD"]:
        h1 = [e for e in ev if ts_ev[e[0]] < np.datetime64("2022-01-01")]
        h2 = [e for e in ev if ts_ev[e[0]] >= np.datetime64("2022-01-01")]
        for tag, sub in [("H1", h1), ("H2", h2)]:
            r = frames(df, sub, rows_of)[("V0", lbl_want)]
            print(f"   V0 {lbl_want} {tag}: n={r[0]}  WR={r[2]:.1%}  avg={r[3]*100:+.3f}%")
