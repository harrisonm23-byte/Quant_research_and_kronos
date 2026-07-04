"""Case study: SPY session of 2026-07-01 across 5m/15m/30m/1h.

Reconstructs the day event-by-event and checks it against every statistical
finding from this project (top timing, contraction, volume, multi-TF state,
which strategy signals fired).
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
DAY = pd.Timestamp("2026-07-01").date()


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


df = pd.read_csv(os.path.join(OUT, "SPY_5m_full.csv"))
df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
df = df.sort_values("ts").reset_index(drop=True)

# keep last ~10 sessions for indicator warmup, RTH only for indicators
rth = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].copy()
rth = rth.reset_index(drop=True)
c = rth["close"]
rth["rsi14"] = wilder_rsi(c)
mid = c.rolling(20).mean()
sd = c.rolling(20).std()
rth["bb_up"] = mid + 2 * sd
rth["bb_lo"] = mid - 2 * sd
rth["day"] = rth["ts"].dt.date
pv = rth["vwap"] * rth["volume"]
rth["svwap"] = pv.groupby(rth["day"]).cumsum() / rth["volume"].groupby(rth["day"]).cumsum()
rth["vol_avg20"] = rth["volume"].rolling(20).mean()

d5 = rth[rth["day"] == DAY].reset_index(drop=True)
prev = rth[rth["day"] < DAY]
prev_day = prev[prev["day"] == prev["day"].max()]
print(f"=== SPY 2026-07-01 RTH: {len(d5)} 5m bars ===")
print(f"open {d5['open'].iloc[0]:.2f}  high {d5['high'].max():.2f}  "
      f"low {d5['low'].min():.2f}  close {d5['close'].iloc[-1]:.2f}  "
      f"prev close {prev_day['close'].iloc[-1]:.2f}")

hi_i = d5["high"].idxmax()
lo_i = d5["low"].idxmin()
print(f"HIGH of day {d5['high'][hi_i]:.2f} in bar {d5['ts'][hi_i].strftime('%H:%M')}-"
      f"{(d5['ts'][hi_i] + pd.Timedelta(minutes=5)).strftime('%H:%M')} ET")
print(f"LOW  of day {d5['low'][lo_i]:.2f} in bar {d5['ts'][lo_i].strftime('%H:%M')}-"
      f"{(d5['ts'][lo_i] + pd.Timedelta(minutes=5)).strftime('%H:%M')} ET")

# ---- event timeline: BB breaks, VWAP crosses, RSI extremes, volume spikes ----
print("\n--- Event timeline (5m, RTH-computed indicators) ---")
above_vwap_prev = None
for i, r in d5.iterrows():
    t = r["ts"].strftime("%H:%M")
    events = []
    if r["close"] >= r["bb_up"]:
        events.append(f"close ABOVE upper BB (rsi={r['rsi14']:.0f})")
    if r["close"] <= r["bb_lo"]:
        events.append(f"close BELOW lower BB (rsi={r['rsi14']:.0f})")
    if r["rsi14"] >= 75:
        events.append(f"RSI14={r['rsi14']:.0f}")
    if r["rsi14"] <= 25:
        events.append(f"RSI14={r['rsi14']:.0f}")
    av = r["close"] > r["svwap"]
    if above_vwap_prev is not None and av != above_vwap_prev:
        events.append("crossed " + ("ABOVE" if av else "BELOW") + f" VWAP ({r['svwap']:.2f})")
    above_vwap_prev = av
    if r["vol_avg20"] and r["volume"] > 3 * r["vol_avg20"]:
        events.append(f"VOLUME {r['volume']/r['vol_avg20']:.1f}x avg "
                      f"({'red' if r['close']<r['open'] else 'green'} bar {r['open']:.2f}->{r['close']:.2f})")
    if events:
        print(f"  {t}  " + "; ".join(events))

# ---- claims check ----
print("\n--- Findings check ---")
# 1. top timing
top_end = d5["ts"][hi_i] + pd.Timedelta(minutes=5)
print(f"1. Top printed in bar ending {top_end.strftime('%H:%M')} "
      f"(minute {top_end.minute:02d}) — 'just after round time' zone is :35-:45")
# 2. contraction into top
if hi_i >= 6:
    last3 = (d5["high"] - d5["low"])[hi_i - 3:hi_i].mean()
    prior = (d5["high"] - d5["low"])[hi_i - 6:hi_i - 3].mean()
    print(f"2. Range into top: last3/prior3 = {last3/prior:.2f} "
          f"({'contracting' if last3 < prior else 'expanding'})")
# 3. volume at/after top vs before
if hi_i >= 6 and hi_i + 3 < len(d5):
    vb = d5["volume"][hi_i - 6:hi_i].mean()
    print(f"3. Volume: top bar {d5['volume'][hi_i]/vb:.2f}x prior-6 avg, "
          f"2 bars after {d5['volume'][hi_i+1:hi_i+3].mean()/vb:.2f}x")
# 4. biggest 5m drop of the day
d5["ret"] = d5["close"] / d5["close"].shift(1) - 1
worst = d5["ret"].idxmin()
print(f"4. Sharpest 5m drop: {d5['ret'][worst]:+.2%} in bar "
      f"{d5['ts'][worst].strftime('%H:%M')}-{(d5['ts'][worst]+pd.Timedelta(minutes=5)).strftime('%H:%M')} ET, "
      f"volume {d5['volume'][worst]/d5['vol_avg20'][worst]:.1f}x avg")

# ---- higher TF state at the top ----
def agg(df5, step):
    minutes = (df5["ts"].dt.hour * 60 + df5["ts"].dt.minute) - (9 * 60 + 30)
    grp = np.minimum(minutes // step, (390 // step) - 1)
    key = df5["day"].astype(str) + "_" + grp.astype(str)
    g = df5.groupby(key, sort=False)
    out = pd.DataFrame({"ts": g["ts"].first(), "open": g["open"].first(),
                        "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last(), "volume": g["volume"].sum()})
    return out.reset_index(drop=True).sort_values("ts").reset_index(drop=True)


print("\n--- Higher-timeframe state at the daily high ---")
for step, name in [(15, "15m"), (30, "30m"), (60, "1h")]:
    a = agg(rth, step)
    a["rsi14"] = wilder_rsi(a["close"])
    m = a["close"].rolling(20).mean()
    s = a["close"].rolling(20).std()
    a["pctb"] = (a["close"] - (m - 2 * s)) / (4 * s)
    before_top = a[a["ts"] <= d5["ts"][hi_i]]
    r = before_top.iloc[-1]
    print(f"  {name:>3s}: RSI14={r['rsi14']:.0f}  %B={r['pctb']:.2f}  at {r['ts'].strftime('%H:%M')}")

# ---- daily context and which strategies signal ----
daily = pd.read_csv(os.path.join(OUT, "SPY_daily.csv"), parse_dates=["date"])
qdaily = pd.read_csv(os.path.join(OUT, "QQQ_daily.csv"), parse_dates=["date"])
for nm, dd in [("SPY", daily), ("QQQ", qdaily)]:
    dd = dd.sort_values("date").reset_index(drop=True)
    r = dd.iloc[-1]
    assert r["date"].date() == DAY, r["date"]
    rng = r["high"] - r["low"]
    ibs = (r["close"] - r["low"]) / rng if rng > 0 else 0.5
    lc5 = dd["close"].rolling(5).min().iloc[-1]
    dr = dd["close"].pct_change().iloc[-1]
    volx = (dd["volume"] / dd["volume"].rolling(20).mean()).iloc[-1]
    cc = dd["close"]
    rsi2 = wilder_rsi(cc, 2).iloc[-1]
    sma20 = cc.rolling(20).mean().iloc[-1]
    weekday = r["date"].weekday()
    print(f"\n--- {nm} daily bar 2026-07-01: O {r['open']:.2f} H {r['high']:.2f} "
          f"L {r['low']:.2f} C {r['close']:.2f} ({dr:+.2%}) ---")
    print(f"  IBS={ibs:.2f}  close{'<=' if r['close']<=lc5 else '>'}5d-low  "
          f"RSI2={rsi2:.0f}  vol={volx:.2f}x  dist20={(r['close']/sma20-1):+.1%}  "
          f"weekday={'Mon Tue Wed Thu Fri'.split()[weekday]}")
    sigs = []
    if ibs < 0.20:
        sigs.append("IBS<0.20 LONG")
    if r["close"] <= lc5 and ibs < 0.25:
        sigs.append("5DayLow+IBS LONG")
    if weekday == 0 and r["close"] < r["open"]:
        sigs.append("TT-A LONG")
    if sigs:
        veto = " — but VOLUME VETO (>1.2x)" if volx > 1.2 else ""
        print(f"  SIGNALS for next open: {', '.join(sigs)}{veto}")
    else:
        print("  no long signals for next open")
