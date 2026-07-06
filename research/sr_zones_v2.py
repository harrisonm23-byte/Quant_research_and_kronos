"""S/R Zones v2 — SPY 5m, 2016-2026 (per approved plan).

Zones: clustered pivots (K=6 confirmed), VALID after >=2 pivots >=60min apart,
width = +/- max(0.05%, 0.20*ATR20), live <=5 sessions, retired 2h after break
(kept 4h for retest tracking). Reference class: prior-day H/L/C.
Events: TEST (race +/-0.25%, 90min) | BREAK (close beyond far edge, continuation
race) | RETEST after break (role-flip test).
Phase D: indicator context recorded at every event; hold-vs-break splits.
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
K = 6                 # pivot arm (30 min)
CLUSTER = 0.001       # 0.10% pivot clustering
MINSEP = 12           # 2nd pivot must be >=60 min after 1st
TGT = 0.0025          # race target 0.25%
HORIZON = 18          # 90 min
OUTSIDE = 6           # bars outside before a new touch counts
BPD = 78              # bars/day

df = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
df["day"] = df["ts"].dt.date
c = df["close"]
h_, l_ = df["high"], df["low"]
tr = np.maximum(h_ - l_, np.maximum((h_ - c.shift(1)).abs(), (l_ - c.shift(1)).abs()))
df["atr20"] = tr.rolling(20).mean()
d = c.diff()
ag = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
al = (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
df["rsi14"] = 100 - 100 / (1 + ag / al)
df["sma9"] = c.rolling(9).mean()
df["sma20"] = c.rolling(20).mean()
df["ema9"] = c.ewm(span=9, adjust=False).mean()
pv = df["vwap"] * df["volume"]
df["svwap"] = pv.groupby(df["day"]).cumsum() / df["volume"].groupby(df["day"]).cumsum()
df["vol20"] = df["volume"].rolling(20).mean()

o = df["open"].values; hh = df["high"].values; ll = df["low"].values; cl = df["close"].values
atr = df["atr20"].values; rsi = df["rsi14"].values
sma9 = df["sma9"].values; sma20 = df["sma20"].values; ema9 = df["ema9"].values
vw = df["svwap"].values; volx = (df["volume"] / df["vol20"]).values
day = df["day"].values
n = len(df)
rows_of = {}
for i, dy in enumerate(day):
    rows_of.setdefault(dy, []).append(i)
days = sorted(rows_of)


def race(i, direction):
    """direction +1: win = +0.25% first. Same-day only. 0 = undecided/ambiguous."""
    px = cl[i]
    up, dn = px * (1 + TGT), px * (1 - TGT)
    de = rows_of[day[i]][-1]
    for j in range(i + 1, min(i + HORIZON + 1, de + 1)):
        u, dwn = hh[j] >= up, ll[j] <= dn
        if u and dwn:
            return 0
        if u:
            return direction
        if dwn:
            return -direction
    return 0


def ctx(i):
    return dict(above_vwap=cl[i] > vw[i], sma9gt20=sma9[i] > sma20[i],
                ema9gt_sma9=ema9[i] > sma9[i], rsi=rsi[i], volx=volx[i])


events = []   # dicts: kind, role, touch, age_h, out, ctx, ts

class Zone:
    __slots__ = ("center", "pivots", "valid", "born", "touches", "outside",
                 "side", "broken", "break_bar", "retested", "cls", "last_piv")
    def __init__(self, center, i, cls="formed"):
        self.center = center
        self.pivots = [i]
        self.valid = cls == "ref"
        self.born = i if cls == "ref" else None
        self.touches = 0
        self.outside = 99
        self.side = None
        self.broken = None
        self.break_bar = None
        self.retested = False
        self.cls = cls
        self.last_piv = i


zones = []
prev_day_ref = {}
for k_ in range(1, len(days)):
    r = rows_of[days[k_ - 1]]
    prev_day_ref[days[k_]] = (max(hh[j] for j in r), min(ll[j] for j in r), cl[r[-1]])

WARM = np.datetime64("2016-06-01")
ts_arr = df["ts"].values
cur_day = None
for i in range(30, n):
    if day[i] != cur_day:
        cur_day = day[i]
        zones = [z for z in zones if z.cls == "formed" and i - z.last_piv < 5 * BPD]
        if cur_day in prev_day_ref:
            for lv in prev_day_ref[cur_day]:
                zones.append(Zone(lv, i, cls="ref"))
                zones[-1].born = i
        zones = zones[-25:]
    if ts_arr[i] < WARM or math.isnan(atr[i]):
        continue
    # confirm pivots at j = i-K
    j = i - K
    if j - K >= 0 and day[j - K] == day[i]:
        for is_low, val in ((True, ll[j]), (False, hh[j])):
            ext = (val == min(ll[j - K:j + K + 1])) if is_low else (val == max(hh[j - K:j + K + 1]))
            if not ext:
                continue
            placed = False
            for z in zones:
                if z.cls == "formed" and abs(z.center - val) / val < CLUSTER:
                    if not z.valid and (j - z.pivots[-1]) >= MINSEP:
                        z.valid = True
                        z.born = i
                    z.pivots.append(j)
                    z.center = float(np.mean([ll[p] if ll[p] <= z.center else hh[p] for p in z.pivots[-4:]]))
                    z.last_piv = i
                    placed = True
                    break
            if not placed:
                zones.append(Zone(val, j))
    # event scan
    px = cl[i]
    w = max(0.0005 * px, 0.20 * atr[i])
    for z in zones:
        if not z.valid:
            continue
        top, bot = z.center + w, z.center - w
        in_zone = ll[i] <= top and hh[i] >= bot
        # retest after break
        if z.broken and not z.retested and z.break_bar and i - z.break_bar <= 48:
            if in_zone and z.outside >= OUTSIDE:
                role = "flip_res" if z.broken == "down" else "flip_sup"
                out = race(i, -1 if z.broken == "down" else +1)
                if out != 0:
                    age_h = (i - z.born) / 12
                    events.append(dict(kind="RETEST", role=role, touch=z.touches,
                                       age=age_h, out=out, i=i, **ctx(i)))
                z.retested = True
                z.outside = 0
                continue
        if z.broken:
            z.outside = 0 if in_zone else z.outside + 1
            continue
        if in_zone and z.outside >= OUTSIDE and z.side in ("above", "below"):
            role = "support" if z.side == "above" else "resist"
            direction = +1 if role == "support" else -1
            out = race(i, direction)
            z.touches += 1
            if out != 0:
                age_h = (i - z.born) / 12
                events.append(dict(kind="TEST", role=role, cls=z.cls,
                                   touch=min(z.touches + 1, 4), age=age_h, out=out, i=i, **ctx(i)))
            z.outside = 0
        elif not in_zone:
            z.outside += 1
            new_side = "above" if cl[i] > top else ("below" if cl[i] < bot else z.side)
            # break check: close decisively beyond far edge after being tested
            if z.side == "above" and cl[i] < bot - 0.0005 * px:
                out = race(i, -1)
                if out != 0:
                    events.append(dict(kind="BREAK", role="sup_break", cls=z.cls,
                                       touch=z.touches, age=(i - z.born) / 12, out=out, i=i, **ctx(i)))
                z.broken = "down"; z.break_bar = i
            elif z.side == "below" and cl[i] > top + 0.0005 * px:
                out = race(i, +1)
                if out != 0:
                    events.append(dict(kind="BREAK", role="res_break", cls=z.cls,
                                       touch=z.touches, age=(i - z.born) / 12, out=out, i=i, **ctx(i)))
                z.broken = "up"; z.break_bar = i
            z.side = new_side
        else:
            z.outside = 0

E = pd.DataFrame(events)
E["win"] = E["out"] > 0
E["half"] = np.where(pd.to_datetime(df["ts"].dt.tz_localize(None)).iloc[E["i"]].values
                     < np.datetime64("2022-01-01"), "H1", "H2")
print(f"events: {len(E)}  (TEST {sum(E.kind=='TEST')}, BREAK {sum(E.kind=='BREAK')}, "
      f"RETEST {sum(E.kind=='RETEST')})")

base = []
for i in range(100, n - 20, 150):
    r_ = race(i, +1)
    if r_ != 0:
        base.append(r_ > 0)
base = np.array(base)
print(f"baseline P(+0.25% first): {base.mean():.1%} (n={len(base)})\n")

print("=== TEST events (does the zone hold?) ===")
for (kind, role), g in E[E.kind == "TEST"].groupby(["kind", "role"]):
    bl = base.mean() if role == "support" else 1 - base.mean()
    print(f"{role:<9s} n={len(g):>5d}  P(hold)={g['win'].mean():.1%} (base {bl:.1%})  "
          f"H1={g[g.half=='H1']['win'].mean():.1%} H2={g[g.half=='H2']['win'].mean():.1%}")
    for t, gg in g.groupby("touch"):
        if len(gg) >= 50:
            print(f"    touch {t}: n={len(gg):>5d}  P(hold)={gg['win'].mean():.1%}")
    for lo_a, hi_a, lbl in [(0, 2, "age<2h"), (2, 6, "2-6h"), (6, 999, ">6h")]:
        gg = g[(g.age >= lo_a) & (g.age < hi_a)]
        if len(gg) >= 50:
            print(f"    {lbl:<7s}: n={len(gg):>5d}  P(hold)={gg['win'].mean():.1%}")

print("\n=== BREAK events (does the break continue?) ===")
for role, g in E[E.kind == "BREAK"].groupby("role"):
    bl = 1 - base.mean() if role == "sup_break" else base.mean()
    print(f"{role:<10s} n={len(g):>5d}  P(continue)={g['win'].mean():.1%} (base {bl:.1%})  "
          f"H1={g[g.half=='H1']['win'].mean():.1%} H2={g[g.half=='H2']['win'].mean():.1%}")

print("\n=== RETEST after break (does the role-flip hold?) ===")
for role, g in E[E.kind == "RETEST"].groupby("role"):
    print(f"{role:<10s} n={len(g):>5d}  P(flip holds)={g['win'].mean():.1%}")

print("\n=== PHASE D: indicator context splits ===")
for kind, role in [("TEST", "support"), ("TEST", "resist"), ("BREAK", "sup_break"), ("BREAK", "res_break")]:
    g = E[(E.kind == kind) & (E.role == role)]
    if len(g) < 100:
        continue
    print(f"\n{kind} {role} (n={len(g)}, base P={g['win'].mean():.1%}):")
    for name, mask in [("above VWAP", g.above_vwap), ("SMA9>SMA20", g.sma9gt20),
                       ("EMA9>SMA9", g.ema9gt_sma9), ("RSI<35", g.rsi < 35),
                       ("RSI>50", g.rsi > 50), ("quiet vol<=1.2", g.volx <= 1.2),
                       ("loud vol>1.5", g.volx > 1.5)]:
        a, b = g[mask], g[~mask]
        if len(a) >= 30 and len(b) >= 30:
            print(f"  {name:<15s} TRUE {a['win'].mean():>6.1%} (n={len(a):>5d}) | "
                  f"FALSE {b['win'].mean():>6.1%} (n={len(b):>5d})  d={a['win'].mean()-b['win'].mean():+.1%}")
