"""Encode R1-R4 from the July-1 case study, scan 10y of SPY 5m, and split
outcomes by higher-timeframe companion indicators.

R1 fade : >=2nd RSI-overbought touch of day + high stalls under round $5 level,
          above VWAP -> short next open
R2 long : 15m#1 red, 15m#2 closes above #1 open, close>VWAP at 10:00 -> long
R3 fade : post-decline retest, EMA9/SMA9 compressed just above VWAP,
          EMA9 crosses below SMA9 -> short next open
R4 cont : close<VWAP & close<lowerBB & vol>=2.5x avg -> short next open
Outcomes at higher-TF checkpoints + EOD; splits by 15m/30m/1h/daily state.
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
d5["rsi14"] = wilder_rsi(c)
d5["ema9"] = c.ewm(span=9, adjust=False).mean()
d5["sma9"] = c.rolling(9).mean()
mid = c.rolling(20).mean()
sd = c.rolling(20).std()
d5["bb_lo"] = mid - 2 * sd
d5["bb_up"] = mid + 2 * sd
d5["day"] = d5["ts"].dt.date
pv = d5["vwap"] * d5["volume"]
d5["svwap"] = pv.groupby(d5["day"]).cumsum() / d5["volume"].groupby(d5["day"]).cumsum()
d5["vol20"] = d5["volume"].rolling(20).mean()

ts = d5["ts"].values
o = d5["open"].values
h = d5["high"].values
lo_ = d5["low"].values
cl = d5["close"].values
vw = d5["svwap"].values
ema9 = d5["ema9"].values
sma9 = d5["sma9"].values
rsi = d5["rsi14"].values
bblo = d5["bb_lo"].values
volx = (d5["volume"] / d5["vol20"]).values
day_arr = d5["day"].values
tod_end = (d5["ts"].dt.hour * 60 + d5["ts"].dt.minute + 5).values
n = len(d5)

# day boundaries
day_start = {}
day_end = {}
for i, dy in enumerate(day_arr):
    if dy not in day_start:
        day_start[dy] = i
    day_end[dy] = i

# ---------- higher-TF frames ----------
def aggregate(step):
    minutes = (d5["ts"].dt.hour * 60 + d5["ts"].dt.minute) - 570
    grp = np.minimum(minutes // step, (390 // step) - 1)
    key = d5["day"].astype(str) + "_" + grp.astype(str)
    g = d5.groupby(key, sort=False)
    a = pd.DataFrame({"ts": g["ts"].first(), "close": g["close"].last(),
                      "end": g["ts"].last() + pd.Timedelta(minutes=5)})
    a = a.reset_index(drop=True).sort_values("ts").reset_index(drop=True)
    a["rsi14"] = wilder_rsi(a["close"])
    m = a["close"].rolling(20).mean()
    s = a["close"].rolling(20).std()
    a["pctb"] = (a["close"] - (m - 2 * s)) / (4 * s)
    e12 = a["close"].ewm(span=12, adjust=False).mean()
    e26 = a["close"].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    a["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    return a


TF = {}
for step, name in [(15, "15m"), (30, "30m"), (60, "1h")]:
    a = aggregate(step)
    TF[name] = (a["end"].values, a[["rsi14", "pctb", "hist"]].values)

daily = pd.read_csv(os.path.join(OUT, "SPY_daily.csv"), parse_dates=["date"])
dc = daily["close"]
dm = dc.rolling(20).mean()
ds = dc.rolling(20).std()
daily["pctb"] = (dc - (dm - 2 * ds)) / (4 * ds)
daily["rsi14"] = wilder_rsi(dc)
d_dates = daily["date"].values
d_feat = daily[["pctb", "rsi14"]].values


def tf_state(sig_ts):
    out = {}
    for name, (ends, feats) in TF.items():
        k = np.searchsorted(ends, sig_ts, side="right") - 1
        out[name] = feats[k] if k >= 0 else (np.nan,) * 3
    dayn = pd.Timestamp(sig_ts).tz_localize(None).normalize()
    kd = np.searchsorted(d_dates, np.datetime64(dayn)) - 1
    out["daily"] = d_feat[kd] if kd >= 0 else (np.nan, np.nan)
    return out


# ---------- signal scans ----------
WARM = np.datetime64("2016-06-01")
sig_lists = {"R1_fade": [], "R2_long": [], "R3_fade": [], "R4_cont": []}

cur_day = None
touch_count = 0
in_touch = False
last_sig = {k: -99 for k in sig_lists}
for i in range(1, n):
    if ts[i] < WARM or math.isnan(rsi[i]) or math.isnan(sma9[i]):
        continue
    dy = day_arr[i]
    if dy != cur_day:
        cur_day = dy
        touch_count = 0
        in_touch = False
    # RSI touch counting
    if rsi[i] >= 65 and not in_touch:
        touch_count += 1
        in_touch = True
    elif rsi[i] < 58:
        in_touch = False
    ds_i = day_start[dy]
    day_hi = h[ds_i:i + 1].max()

    px = cl[i]
    # R1: >=2nd touch, stall under round $5, above VWAP
    round_lvl = math.ceil(day_hi / 5) * 5
    if (touch_count >= 2 and in_touch and rsi[i] >= 65 and px > vw[i]
            and 0 < round_lvl - day_hi <= 0.0015 * px and i - last_sig["R1_fade"] > 12):
        sig_lists["R1_fade"].append(i)
        last_sig["R1_fade"] = i
    # R2: at the 9:55-10:00 bar (6th bar of day)
    if i - ds_i == 5:
        o1 = o[ds_i]
        c1 = cl[ds_i + 2]
        c2 = cl[i]
        if c1 < o1 and c2 > o1 and c2 > vw[i]:
            sig_lists["R2_long"].append(i)
    # R3: compression + bearish cross above VWAP, post-decline
    spread = max(ema9[i], sma9[i]) - min(ema9[i], sma9[i])
    above = min(ema9[i], sma9[i]) - vw[i]
    crossed = ema9[i] < sma9[i] and ema9[i - 1] >= sma9[i - 1]
    if (crossed and spread < 0.0003 * px and 0 < above < 0.0015 * px
            and px > vw[i] and px < day_hi * (1 - 0.002) and i - last_sig["R3_fade"] > 12):
        sig_lists["R3_fade"].append(i)
        last_sig["R3_fade"] = i
    # R4: breakdown confirmation
    if (px < vw[i] and px < bblo[i] and volx[i] >= 2.5 and i - last_sig["R4_cont"] > 12):
        sig_lists["R4_cont"].append(i)
        last_sig["R4_cont"] = i


def next_boundary(i, step_min):
    for j in range(i + 2, min(i + 40, n)):
        if day_arr[j] != day_arr[i]:
            return None
        if tod_end[j] % step_min == 0:
            return j
    return None


def outcomes(i, side):
    """side=+1 long, -1 short; entry next 5m open."""
    if i + 2 >= n or day_arr[i + 1] != day_arr[i]:
        return None
    e = o[i + 1]
    res = {}
    j15 = next_boundary(i, 15)
    j30 = next_boundary(i, 30)
    res["15m_ck"] = side * (cl[j15] / e - 1) if j15 else np.nan
    res["30m_ck"] = side * (cl[j30] / e - 1) if j30 else np.nan
    if j30 and j30 + 6 < n and day_arr[j30 + 6] == day_arr[i]:
        res["30m_next"] = side * (cl[j30 + 6] / e - 1)
    else:
        res["30m_next"] = np.nan
    de = day_end[day_arr[i]]
    res["EOD"] = side * (cl[de] / e - 1)
    if side < 0:
        w = lo_[i + 1:min(i + 13, de + 1)]
        res["MFE60"] = (e / w.min() - 1) if len(w) else np.nan
    else:
        w = h[i + 1:min(i + 13, de + 1)]
        res["MFE60"] = (w.max() / e - 1) if len(w) else np.nan
    return res


SPLITS = {
    "R1_fade": [("30m RSI>65", lambda s: s["30m"][0] > 65),
                ("1h %B>0.9", lambda s: s["1h"][1] > 0.9),
                ("15m MACD hist<0", lambda s: s["15m"][2] < 0),
                ("daily %B>0.8", lambda s: s["daily"][0] > 0.8)],
    "R2_long": [("30m RSI<35", lambda s: s["30m"][0] < 35),
                ("daily %B<0.3", lambda s: s["daily"][0] < 0.3),
                ("1h %B<0.2", lambda s: s["1h"][1] < 0.2),
                ("15m MACD hist>0", lambda s: s["15m"][2] > 0)],
    "R3_fade": [("30m RSI>60", lambda s: s["30m"][0] > 60),
                ("15m MACD hist<0", lambda s: s["15m"][2] < 0),
                ("1h %B>0.8", lambda s: s["1h"][1] > 0.8),
                ("daily %B>0.8", lambda s: s["daily"][0] > 0.8)],
    "R4_cont": [("15m MACD hist<0", lambda s: s["15m"][2] < 0),
                ("30m RSI<40", lambda s: s["30m"][0] < 40),
                ("1h %B<0.3", lambda s: s["1h"][1] < 0.3),
                ("daily %B>0.8", lambda s: s["daily"][0] > 0.8)],
}
SIDE = {"R1_fade": -1, "R2_long": +1, "R3_fade": -1, "R4_cont": -1}

for rule, idxs in sig_lists.items():
    side = SIDE[rule]
    rows = []
    for i in idxs:
        oc = outcomes(i, side)
        if oc is None:
            continue
        st = tf_state(ts[i])
        rows.append((i, oc, st))
    print(f"\n=== {rule}  ({len(rows)} signals, {'short' if side<0 else 'long'}) ===")
    if not rows:
        continue
    for ck in ["15m_ck", "30m_ck", "30m_next", "EOD", "MFE60"]:
        a = np.array([r[1][ck] for r in rows])
        a = a[~np.isnan(a)]
        if len(a):
            print(f"  {ck:<9s} n={len(a):>4d}  WR={(a>0).mean():>6.1%}  avg={a.mean():>+8.3%}")
    print("  --- higher-TF companion splits (outcome=EOD) ---")
    eod = np.array([r[1]["EOD"] for r in rows])
    for name, fn in SPLITS[rule]:
        mask = np.array([bool(fn(r[2])) if not any(np.isnan(np.atleast_1d(v)).any()
                        for v in r[2].values()) else False for r in rows])
        at, af = eod[mask], eod[~mask]
        at, af = at[~np.isnan(at)], af[~np.isnan(af)]
        if len(at) >= 10 and len(af) >= 10:
            print(f"    {name:<18s} TRUE: n={len(at):>4d} WR={(at>0).mean():>6.1%} avg={at.mean():+.3%}"
                  f"   FALSE: n={len(af):>4d} WR={(af>0).mean():>6.1%} avg={af.mean():+.3%}")
        else:
            print(f"    {name:<18s} n_true={len(at)} (too few for split)")

# July 1 stamps
print("\n=== July 1 higher-TF stamps at each rule's firing (if fired) ===")
for rule, idxs in sig_lists.items():
    for i in idxs:
        if day_arr[i] == pd.Timestamp("2026-07-01").date():
            st = tf_state(ts[i])
            t = pd.Timestamp(ts[i]).strftime("%H:%M")
            print(f"  {rule} at {t} ET: 15m(rsi={st['15m'][0]:.0f},hist={st['15m'][2]:+.2f}) "
                  f"30m(rsi={st['30m'][0]:.0f},%B={st['30m'][1]:.2f}) "
                  f"1h(%B={st['1h'][1]:.2f}) daily(%B={st['daily'][0]:.2f})")
