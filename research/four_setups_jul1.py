"""Verify the user's four observed July-1 setups against 5m data (ET times).

1. ~12:15-12:25 ET: failed break of resistance + Nth touch of RSI-overbought -> reversal
2. ~10:00 ET: 2nd 15m candle closes far above 1st 15m candle's open (opening recovery)
3. ~15:30 ET: EMA9/SMA9/price converge into VWAP from below and fail
4. ~15:50 ET: closes below VWAP + lower BB + MAs cross below VWAP + volume expands
"""
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
rth = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].reset_index(drop=True)
rth["rsi14"] = wilder_rsi(rth["close"])
rth["ema9"] = rth["close"].ewm(span=9, adjust=False).mean()
rth["sma9"] = rth["close"].rolling(9).mean()
rth["day"] = rth["ts"].dt.date
pv = rth["vwap"] * rth["volume"]
rth["svwap"] = pv.groupby(rth["day"]).cumsum() / rth["volume"].groupby(rth["day"]).cumsum()
mid = rth["close"].rolling(20).mean()
sd = rth["close"].rolling(20).std()
rth["bb_lo"] = mid - 2 * sd
d = rth[rth["day"] == DAY].reset_index(drop=True)

# prior day high (resistance)
prev = rth[rth["day"] < DAY]
prev_hi = prev[prev["day"] == prev["day"].max()]["high"].max()

print("--- 1. The top: RSI-touch count and resistance test (12:00-12:30 ET) ---")
print(f"prior-day (Jun 30) high: {prev_hi:.2f}; overhead round number: 750.00")
hi = d["high"].cummax()
touches = []
in_touch = False
for i, r in d.iterrows():
    if r["rsi14"] >= 68 and not in_touch:
        touches.append((r["ts"].strftime("%H:%M"), r["rsi14"], r["high"]))
        in_touch = True
    elif r["rsi14"] < 64:
        in_touch = False
print("RSI14 overbought touches (>=68, reset <64):")
for k, (t, rsi, h) in enumerate(touches, 1):
    print(f"  touch {k}: {t} ET  rsi={rsi:.0f}  bar high={h:.2f}")
top_i = d["high"].idxmax()
print(f"day high 749.43 at 12:20-12:25 ET vs prior-day high {prev_hi:.2f} -> "
      f"{'FAILED to clear' if d['high'].max() <= prev_hi else 'cleared'} "
      f"(shortfall {prev_hi - d['high'].max():+.2f})")

print("\n--- 2. Opening 15m structure (9:30-10:00 ET) ---")
c1 = d.iloc[0:3]   # 9:30-9:45
c2 = d.iloc[3:6]   # 9:45-10:00
o1, c1c = c1["open"].iloc[0], c1["close"].iloc[-1]
o2, c2c = c2["open"].iloc[0], c2["close"].iloc[-1]
print(f"15m #1 (9:30-45): O {o1:.2f} L {c1['low'].min():.2f} C {c1c:.2f}  "
      f"({'red' if c1c<o1 else 'green'})")
print(f"15m #2 (9:45-10:00): O {o2:.2f} C {c2c:.2f}  closes {c2c-o1:+.2f} vs #1 open "
      f"-> {'ENGULFED' if c2c > o1 else 'no'}")
vwap_10 = d["svwap"].iloc[5]
print(f"at 10:00 ET: close {c2c:.2f} vs VWAP {vwap_10:.2f} "
      f"({'above' if c2c > vwap_10 else 'below'})")

print("\n--- 3. Convergence failure (~15:30 ET) ---")
for i in range(66, 74):
    r = d.iloc[i]
    t = r["ts"].strftime("%H:%M")
    spread = max(r["ema9"], r["sma9"], r["svwap"]) - min(r["ema9"], r["sma9"], r["svwap"])
    print(f"  {t}  close={r['close']:.2f}  ema9={r['ema9']:.2f}  sma9={r['sma9']:.2f}  "
          f"vwap={r['svwap']:.2f}  spread={spread:.2f}"
          + ("  <- max convergence" if spread < 0.15 else ""))

print("\n--- 4. Breakdown confirmation (15:40-15:55 ET) ---")
va = d["volume"].rolling(20).mean()
for i in range(74, 78):
    r = d.iloc[i]
    t = r["ts"].strftime("%H:%M")
    marks = []
    if r["close"] < r["svwap"]: marks.append("close<VWAP")
    if r["close"] < r["bb_lo"]: marks.append("close<lowerBB")
    if r["ema9"] < r["svwap"]: marks.append("EMA9<VWAP")
    if r["sma9"] < r["svwap"]: marks.append("SMA9<VWAP")
    vx = r["volume"] / va.iloc[i] if va.iloc[i] else float("nan")
    print(f"  {t}  C={r['close']:.2f}  vol={vx:.1f}x  {'; '.join(marks)}")

# follow-through: extended hours after 16:00
ext = df[(df["ts"].dt.date == DAY) & (df["ts"].dt.time > dtime(15, 55))]
if len(ext):
    print(f"\nafter-hours follow-through: 16:00 close {ext['close'].iloc[0]:.2f} -> "
          f"19:55 {ext['close'].iloc[-1]:.2f} (low {ext['low'].min():.2f})")
entry4 = d["close"].iloc[75]  # 15:45 close as proxy short entry after 15:40 signal
print(f"R4 short from 15:45 close {entry4:.2f}: to RTH close {d['close'].iloc[-1]:.2f} "
      f"({entry4/d['close'].iloc[-1]-1:+.2%}), to AH low {ext['low'].min():.2f} "
      f"({entry4/ext['low'].min()-1:+.2%})")
