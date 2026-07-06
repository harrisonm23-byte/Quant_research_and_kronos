"""Test the 5m exhaustion-top setup and its microstructure claims.

Signal (5m bar, RTH): close >= upper BB(20,2) AND RSI14 > 80 AND close > session VWAP.
De-duplicated: first bar of each cluster (no signal in prior 6 bars).

Claims tested:
 1. Round-time reversal: the local top after the signal prints at :00/:30 marks.
 2. Candles shrink into the top (range contraction), volume expands on reversal.
 3. Win should be scored at higher-TF checkpoints (next 15m/30m close, the 30m
    after that), not the next 5m candle.
 4. Multi-TF confirmation: 30m RSI and prior-day daily %B state change outcomes.
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
rng_seed = np.random.RandomState(42)


def wilder_rsi(close, period):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


# ---------- load 5m ----------
df = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
keep = (df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))
df = df[keep].sort_values("ts").reset_index(drop=True)
c = df["close"]
df["rsi14"] = wilder_rsi(c, 14)
mid = c.rolling(20).mean()
sd = c.rolling(20).std()
df["bb_up"] = mid + 2 * sd
df["day"] = df["ts"].dt.date
pv = df["vwap"] * df["volume"]
df["svwap"] = pv.groupby(df["day"]).cumsum() / df["volume"].groupby(df["day"]).cumsum()
df["range"] = df["high"] - df["low"]

ts = df["ts"].values
o = df["open"].values
h = df["high"].values
l = df["low"].values
cl = df["close"].values
vol = df["volume"].values.astype(float)
rgn = df["range"].values
minute_end = ((df["ts"].dt.hour * 60 + df["ts"].dt.minute + 5) % 60).values
tod_end = (df["ts"].dt.hour * 60 + df["ts"].dt.minute + 5).values  # minutes after midnight
day_arr = df["day"].values
n = len(df)

sig = (df["close"] >= df["bb_up"]) & (df["rsi14"] > 80) & (df["close"] > df["svwap"])
sig = sig.fillna(False).values
# de-dup: first of cluster
sig_idx = []
last = -99
for i in np.flatnonzero(sig):
    if i - last > 6:
        sig_idx.append(i)
    last = i
sig_idx = np.array(sig_idx)
print(f"raw signal bars: {sig.sum()}, de-duplicated clusters: {len(sig_idx)} "
      f"over {df['day'].nunique()} sessions ({len(sig_idx)/df['day'].nunique():.2f}/day)")

W = 12  # 60-minute window for the local top


def local_top(i):
    j_end = min(i + W, n - 1)
    if day_arr[j_end] != day_arr[i]:          # clamp to same session
        j_end = i + int(np.argmax(day_arr[i:j_end + 1] != day_arr[i])) - 1
    win = h[i:j_end + 1]
    if len(win) < 4:
        return None
    return i + int(np.argmax(win))


# ---------- 1. round-time top ----------
def top_minute_dist(indices):
    mins, tods = [], []
    for i in indices:
        t = local_top(i)
        if t is not None:
            mins.append(minute_end[t])
            tods.append(tod_end[t])
    return np.array(mins), np.array(tods)


sig_min, sig_tod = top_minute_dist(sig_idx)
base_sample = rng_seed.choice(np.arange(30, n - 20), size=20000, replace=False)
base_min, _ = top_minute_dist(base_sample)

round_share_sig = np.isin(sig_min, [0, 30]).mean()
round_share_base = np.isin(base_min, [0, 30]).mean()
print(f"\n1. ROUND-TIME TOPS (bar-END minute of the local top within 60min after signal)")
print(f"   tops ending on :00/:30 — signal: {round_share_sig:.1%} (n={len(sig_min)}), "
      f"baseline: {round_share_base:.1%}, uniform: 16.7%")
print("   minute-of-hour histogram (signal tops):")
for m in range(0, 60, 5):
    cnt = (sig_min == m).mean()
    b = (base_min == m).mean()
    print(f"     :{m:02d}  {cnt:>5.1%}  (baseline {b:>5.1%})  {'#' * int(cnt * 200)}")
print("   time-of-day of tops (top half-hours):")
tod_ser = pd.Series(sig_tod // 30 * 30)
top_slots = tod_ser.value_counts(normalize=True).head(6)
for slot, share in top_slots.items():
    print(f"     {int(slot)//60:02d}:{int(slot)%60:02d}  {share:.1%}")

# ---------- 2. contraction into top / volume on reversal ----------
ctr, volx_top, volx_rev = [], [], []
for i in sig_idx:
    t = local_top(i)
    if t is None or t < i + 3 or t + 2 >= n or day_arr[t + 2] != day_arr[t]:
        continue
    pre = rgn[max(t - 6, i):t - 2].mean() if t - 3 > i else np.nan
    last3 = rgn[t - 3:t].mean()
    if pre and not math.isnan(pre) and pre > 0:
        ctr.append(last3 / pre)
    vbase = vol[max(t - 6, i):t].mean()
    if vbase > 0:
        volx_top.append(vol[t] / vbase)
        volx_rev.append(vol[t + 1:t + 3].mean() / vbase)
ctr, volx_top, volx_rev = map(np.array, (ctr, volx_top, volx_rev))
print(f"\n2. INTO-THE-TOP MICROSTRUCTURE (n={len(ctr)})")
print(f"   range of last 3 bars into top vs prior bars: median ratio {np.median(ctr):.2f} "
      f"({(ctr < 1).mean():.0%} of cases contracting)")
print(f"   volume at top bar vs prior 6-bar avg: median {np.median(volx_top):.2f}x")
print(f"   volume of 2 bars AFTER top vs prior avg: median {np.median(volx_rev):.2f}x "
      f"({(volx_rev > 1).mean():.0%} of cases expanding)")

# ---------- 3. higher-TF checkpoint scoring (short from signal) ----------
def next_boundary_close(i, step_min):
    """Index of first bar (after entry bar i+1) whose END is a step_min boundary."""
    for j in range(i + 1, min(i + 40, n)):
        if day_arr[j] != day_arr[i]:
            return None
        if tod_end[j] % step_min == 0 and j > i + 1:
            return j
    return None


chk = {"next 5m close": [], "next 15m close": [], "next 30m close": [],
       "30m after that": [], "+60min": [], "+120min": []}
mfe = {30: [], 60: [], 120: []}
for i in sig_idx:
    if i + 25 >= n or day_arr[i + 25] != day_arr[i]:
        continue
    entry = o[i + 1]
    chk["next 5m close"].append(entry / cl[i + 1] - 1)
    j15 = next_boundary_close(i, 15)
    j30 = next_boundary_close(i, 30)
    if j15: chk["next 15m close"].append(entry / cl[j15] - 1)
    if j30:
        chk["next 30m close"].append(entry / cl[j30] - 1)
        j30b = j30 + 6
        if j30b < n and day_arr[j30b] == day_arr[i]:
            chk["30m after that"].append(entry / cl[j30b] - 1)
    chk["+60min"].append(entry / cl[i + 13] - 1)
    chk["+120min"].append(entry / cl[i + 25] - 1)
    for wmin, arr in mfe.items():
        wbars = wmin // 5
        arr.append(entry / l[i + 1:i + 1 + wbars].min() - 1)

print(f"\n3. SHORT FROM SIGNAL, SCORED AT HIGHER-TF CHECKPOINTS (entry next 5m open)")
print(f"   {'checkpoint':<16s} {'n':>5s} {'WR':>6s} {'avg':>8s} {'med':>8s}")
for k, arr in chk.items():
    a = np.array(arr)
    print(f"   {k:<16s} {len(a):>5d} {(a>0).mean():>6.1%} {a.mean():>8.3%} {np.median(a):>8.3%}")
print("   max favorable excursion (deepest low below entry within window):")
for wmin, arr in mfe.items():
    a = np.array(arr)
    print(f"     within {wmin:>3d}min: P(>=0.10%)={ (a>=0.001).mean():.0%}  "
          f"P(>=0.20%)={(a>=0.002).mean():.0%}  P(>=0.30%)={(a>=0.003).mean():.0%}")

# ---------- 4. multi-TF confirmation splits ----------
m30 = pd.read_csv(os.path.join(OUT, "SPY_30m_full.csv"))
m30["ts"] = pd.to_datetime(m30["timestamps"]).dt.tz_convert(NY)
keep = (m30["ts"].dt.time >= dtime(9, 30)) & (m30["ts"].dt.time <= dtime(15, 30))
m30 = m30[keep].sort_values("ts").reset_index(drop=True)
m30["rsi14"] = wilder_rsi(m30["close"], 14)
m30_end = m30["ts"] + pd.Timedelta(minutes=30)

daily = pd.read_csv(os.path.join(OUT, "SPY_daily.csv"), parse_dates=["date"])
dc = daily["close"]
dmid = dc.rolling(20).mean()
dsd = dc.rolling(20).std()
daily["pctb"] = (dc - (dmid - 2 * dsd)) / (4 * dsd)

m30_ends = m30_end.values
m30_rsi = m30["rsi14"].values
d_dates = daily["date"].values
d_pctb = daily["pctb"].values

sig_ts = ts[sig_idx]
res30 = []
for i in sig_idx:
    j30 = next_boundary_close(i, 30)
    if j30 is None or i + 1 >= n:
        res30.append(np.nan)
    else:
        res30.append(o[i + 1] / cl[j30] - 1)
res30 = np.array(res30)

k = np.searchsorted(m30_ends, sig_ts, side="right") - 1
rsi30_at_sig = np.where(k >= 0, m30_rsi[k], np.nan)
sig_days = pd.to_datetime(pd.Series(sig_ts)).dt.tz_localize(None).dt.normalize().values
kd = np.searchsorted(d_dates, sig_days) - 1
pctb_prev = np.where(kd >= 0, d_pctb[kd], np.nan)

print(f"\n4. MULTI-TIMEFRAME SPLITS (outcome = short return at next 30m close)")
ok = ~np.isnan(res30)
for name, mask in [
    ("30m RSI14 > 70", rsi30_at_sig > 70),
    ("30m RSI14 <= 70", rsi30_at_sig <= 70),
    ("prev-day %B > 0.8", pctb_prev > 0.8),
    ("prev-day %B <= 0.8", pctb_prev <= 0.8),
    ("BOTH 30m>70 & day %B>.8", (rsi30_at_sig > 70) & (pctb_prev > 0.8)),
]:
    a = res30[ok & mask]
    if len(a) >= 15:
        print(f"   {name:<26s} n={len(a):>4d}  WR={(a>0).mean():>6.1%}  avg={a.mean():>+8.3%}")
    else:
        print(f"   {name:<26s} n={len(a):>4d}  (too few)")
