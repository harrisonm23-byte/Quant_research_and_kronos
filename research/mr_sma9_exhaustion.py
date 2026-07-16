#!/usr/bin/env python3
"""Formalized put-side exhaustion signal on SPY 5m.

SETUP (stretch / one-sided trend):
  1. close >= upper Bollinger (or %B >= 0.95)
  2. close far above VWAP (stretch threshold)
  3. lower Bollinger band itself is above VWAP
     -> entire BB envelope elevated above the session mean
  4. RSI near top (>= rsi_hi)

STRUCTURE (SMA9 behavior — the new piece):
  A. streak_above  = consecutive closes STRICTLY ABOVE SMA9
     (long streak = one-sided grind; drawdown more likely)
  B. bounce_count  = times price wicked to/near SMA9 then closed back above
     without a close below (rejected off SMA9)
  C. bar_len_avg   = avg (high-low)/close over the streak (extension size)

TRIGGER (turn):
  First close UNDER SMA9 after a qualifying setup (or after long streak_above).
  Optional: MACD hist declining on the break bar.

EXIT (underlying proxy for ~20-25m options scalp):
  +target MFE, touch VWAP, or time stop (5 bars = 25m).

Usage:
  python3 mr_sma9_exhaustion.py
  python3 mr_sma9_exhaustion.py --scan
"""
import argparse
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.00015
HOLD = 5
DEDUP = 6

# --- formal thresholds (tunable) ---
VWAP_STRETCH = 0.0025      # close >= VWAP + 0.25%
PCTB_HI = 0.95             # upper BB region
RSI_HI = 65
STREAK_MIN = 6             # >= 6 closes above SMA9 (~30m of one-sided grind)
BOUNCE_MIN = 2             # >= 2 SMA9 rejects before break
SMA9_TOUCH = 0.0008        # wick within 0.08% of SMA9 counts as "bounce off"
TARGET = 0.0020            # 0.20% underlying take-profit
BB_ABOVE_VWAP = True       # require bb_lo > vwap


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def macd_hist(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return line - line.ewm(span=9, adjust=False).mean()


def load_5m():
    for name in ("SPY_5m_full.csv", "SPY_5m_yf.csv"):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = next(c for c in ("timestamps", "datetime", "ts") if c in df.columns)
            df["ts"] = pd.to_datetime(df[col], utc=True).dt.tz_convert(NY)
            print(f"Loaded {path}: {len(df)} bars")
            break
    else:
        import yfinance as yf
        raw = yf.download("SPY", interval="5m", period="60d",
                          auto_adjust=True, progress=False).reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                           for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        ts_col = "datetime" if "datetime" in raw.columns else raw.columns[0]
        df = raw.rename(columns={ts_col: "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        df.to_csv(os.path.join(OUT, "SPY_5m_yf.csv"), index=False)

    keep = (df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))
    df = df.loc[keep].sort_values("ts").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["day"] = df["ts"].dt.date
    return df


def prep(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_pv = (tp * df["volume"]).groupby(df["day"]).cumsum()
    cum_v = df["volume"].groupby(df["day"]).cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_v
    df["vwap_dist"] = df["close"] / df["vwap"] - 1.0
    df["sma9"] = df["close"].rolling(9).mean()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    df["pctb"] = (df["close"] - df["bb_lo"]) / (4 * sd)
    df["rsi14"] = wilder_rsi(df["close"])
    df["macdh"] = macd_hist(df["close"])
    df["bar_len"] = (df["high"] - df["low"]) / df["close"]
    df["above_sma9"] = df["close"] > df["sma9"]
    df["below_sma9"] = df["close"] < df["sma9"]
    # wick touched SMA9 from above (bounce candidate)
    df["wick_to_sma9"] = (
        (df["low"] <= df["sma9"] * (1 + SMA9_TOUCH))
        & (df["close"] > df["sma9"])
        & (df["open"] > df["sma9"])
    )
    return df


def annotate_streaks(df):
    """Per bar: current consecutive closes above SMA9, bounce count in streak, avg bar len."""
    above = df["above_sma9"].values
    wick = df["wick_to_sma9"].values
    blen = df["bar_len"].values
    day = df["day"].values
    n = len(df)
    streak = np.zeros(n, dtype=int)
    bounces = np.zeros(n, dtype=int)
    avg_len = np.full(n, np.nan)
    s = b = 0
    lens = []
    for i in range(n):
        if i > 0 and day[i] != day[i - 1]:
            s = b = 0
            lens = []
        if above[i]:
            s += 1
            lens.append(blen[i])
            if wick[i]:
                b += 1
        else:
            s = 0
            b = 0
            lens = []
        streak[i] = s
        bounces[i] = b
        avg_len[i] = float(np.mean(lens)) if lens else np.nan
    df["streak_above"] = streak
    df["bounce_count"] = bounces
    df["streak_bar_len"] = avg_len
    return df


def setup_ok(r):
    """Stretch / elevated-envelope conditions at bar r (before or on break)."""
    if not (r.vwap_dist >= VWAP_STRETCH):
        return False
    if not (r.pctb >= PCTB_HI or r.close >= r.bb_up):
        return False
    if BB_ABOVE_VWAP and not (r.bb_lo > r.vwap):
        return False
    if not (r.rsi14 >= RSI_HI):
        return False
    return True


def structure_ok(r, mode="streak_or_bounce"):
    """SMA9 one-sided structure."""
    if mode == "streak":
        return r.streak_above >= STREAK_MIN
    if mode == "bounce":
        return r.bounce_count >= BOUNCE_MIN and r.streak_above >= 3
    # default: either long streak OR enough bounces
    return (r.streak_above >= STREAK_MIN) or (r.bounce_count >= BOUNCE_MIN and r.streak_above >= 4)


def backtest(df, mode="streak_or_bounce", need_macd=False, label=""):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    vwap = df["vwap"].values
    below = df["below_sma9"].values
    above = df["above_sma9"].values
    macdh = df["macdh"].values
    day = df["day"].values
    rows = list(df.itertuples(index=False))
    n = len(df)
    trades, last = [], -999

    for i in range(40, n - HOLD - 1):
        if i - last < DEDUP:
            continue
        # TRIGGER: first close under SMA9
        if not below[i]:
            continue
        if i > 0 and below[i - 1]:  # already under — not first break
            continue
        # look at prior bar (still above) for setup/structure
        prev = rows[i - 1]
        if not prev.above_sma9:
            continue
        if not structure_ok(prev, mode):
            continue
        # setup: either on break bar or within last 3 bars of streak
        setup = setup_ok(prev)
        if not setup:
            # allow setup earlier in streak: scan back up to streak length
            look = min(int(prev.streak_above), 12)
            found = False
            for k in range(1, look + 1):
                if setup_ok(rows[i - k]):
                    found = True
                    break
            if not found:
                continue

        if need_macd and not (macdh[i] < macdh[i - 1]):
            continue

        # PUT entry next open
        entry = o[i + 1] * (1 - SLIP)
        exit_px, reason, held = None, "time", HOLD
        for k in range(1, HOLD + 1):
            j = i + 1 + k - 1
            if j >= n or day[j] != day[i + 1]:
                break
            held = k
            if l[j] <= entry * (1 - TARGET):
                exit_px, reason = entry * (1 - TARGET), "target"
                break
            if l[j] <= vwap[j]:
                exit_px, reason = vwap[j], "vwap"
                break
        if exit_px is None:
            exit_px = c[min(i + HOLD, n - 1)]

        ret = entry / exit_px - 1
        mfe = entry / l[i + 1:i + 1 + held].min() - 1
        trades.append(dict(
            ts=df["ts"].iloc[i], day=day[i],
            streak=int(prev.streak_above), bounces=int(prev.bounce_count),
            bar_len=float(prev.streak_bar_len) if prev.streak_bar_len == prev.streak_bar_len else np.nan,
            stretch=float(prev.vwap_dist), rsi=float(prev.rsi14), pctb=float(prev.pctb),
            bb_lo_above_vwap=bool(prev.bb_lo > prev.vwap),
            entry=entry, exit=exit_px, ret=ret, mfe=mfe,
            held=held, reason=reason,
        ))
        last = i

    tr = pd.DataFrame(trades)
    n_days = df["day"].nunique()
    print(f"\n{label}")
    if not len(tr):
        print("  no trades")
        return tr
    print(f"  n={len(tr)} ({len(tr)/n_days:.2f}/day)")
    print(f"  WR={(tr['ret']>0).mean():.0%}  avg={tr['ret'].mean():+.3%}  med={tr['ret'].median():+.3%}")
    print(f"  MFE med={tr['mfe'].median():+.3%}  "
          f"P(MFE>=0.15%)={(tr['mfe']>=0.0015).mean():.0%}  "
          f"P(MFE>=0.20%)={(tr['mfe']>=0.002).mean():.0%}")
    print(f"  exits: target={(tr['reason']=='target').mean():.0%}  "
          f"vwap={(tr['reason']=='vwap').mean():.0%}  time={(tr['reason']=='time').mean():.0%}")
    print(f"  med streak_above={tr['streak'].median():.0f}  "
          f"med bounces={tr['bounces'].median():.0f}  "
          f"med stretch={tr['stretch'].median():.2%}")
    # split by structure richness
    rich = tr[(tr["streak"] >= STREAK_MIN) & (tr["bounces"] >= BOUNCE_MIN)]
    if len(rich) >= 5:
        print(f"  rich (streak>={STREAK_MIN} & bounces>={BOUNCE_MIN}): "
              f"n={len(rich)} WR={(rich['ret']>0).mean():.0%} avg={rich['ret'].mean():+.3%} "
              f"MFE med={rich['mfe'].median():+.3%}")
    return tr


def scan(df):
    print("\n" + "=" * 72)
    print("SCAN — SMA9 exhaustion state")
    print("=" * 72)
    for _, r in df.tail(12).iterrows():
        flags = []
        if r.vwap_dist >= VWAP_STRETCH:
            flags.append(f"vwap+{r.vwap_dist:.2%}")
        if r.pctb >= PCTB_HI:
            flags.append("upperBB")
        if r.bb_lo > r.vwap:
            flags.append("bbLo>vwap")
        if r.rsi14 >= RSI_HI:
            flags.append(f"RSI{r.rsi14:.0f}")
        if r.streak_above >= STREAK_MIN:
            flags.append(f"streak{r.streak_above}")
        if r.bounce_count >= BOUNCE_MIN:
            flags.append(f"bounce{r.bounce_count}")
        armed = (r.streak_above >= STREAK_MIN or r.bounce_count >= BOUNCE_MIN) and setup_ok(r)
        mark = " *** SETUP ARMED (wait SMA9 break for puts)" if armed else ""
        if r.below_sma9 and _ > 0:
            mark = " *** BREAK under SMA9" + mark
        print(f"  {r.ts.strftime('%m-%d %H:%M')} c={r.close:.2f} sma9={r.sma9:.2f} "
              f"vwap={r.vwap:.2f} streak={r.streak_above} bounce={r.bounce_count} "
              f"{'|'.join(flags) if flags else '-'}{mark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", action="store_true")
    args = p.parse_args()
    df = annotate_streaks(prep(load_5m()))
    print(f"RTH {len(df)} bars | {df['day'].nunique()} days | "
          f"{df['ts'].iloc[0].date()} -> {df['ts'].iloc[-1].date()}")
    print("FORMAL PUT SIGNAL")
    print(f"  setup: upperBB + VWAP stretch>={VWAP_STRETCH:.2%} + bb_lo>vwap + RSI>={RSI_HI}")
    print(f"  structure: streak_above>={STREAK_MIN} OR bounces>={BOUNCE_MIN}")
    print(f"  trigger: first close under SMA9")
    print(f"  exit: +{TARGET:.2%} / VWAP touch / {HOLD*5}m")

    if args.scan:
        scan(df)
        return

    print("\n" + "=" * 72)
    print("BACKTEST VARIANTS")
    print("=" * 72)
    backtest(df, mode="streak", label="A. streak-only (long one-sided grind)")
    backtest(df, mode="bounce", label="B. bounce-only (SMA9 rejects then break)")
    tr = backtest(df, mode="streak_or_bounce", label="C. streak OR bounce (primary)")
    backtest(df, mode="streak_or_bounce", need_macd=True,
             label="D. primary + MACD hist declining on break")

    if len(tr):
        path = os.path.join(OUT, "mr_sma9_exhaustion_trades.csv")
        tr.to_csv(path, index=False)
        print(f"\nWrote {path}")
    scan(df)


if __name__ == "__main__":
    main()
