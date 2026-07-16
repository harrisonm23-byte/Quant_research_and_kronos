#!/usr/bin/env python3
"""SPY 5m VWAP mean-reversion — matched to manual options scalps.

Idea (from chart examples):
  - Treat the day as one candle; 5m bars are the granular path of that OHLC.
  - When price stretches away from VWAP (yellow) and hugs a Bollinger edge,
    it tends to move back toward VWAP — often only 0.15–0.35%, not 0.5%+.
  - That small underlying move is enough if you buy puts/calls at the turn.

Indicators (Robinhood colors):
  EMA9 blue, SMA9 orange, BB(20,2) light blue, VWAP yellow, RSI14, MACD(12,26,9)

Signal:
  SHORT (puts): close above VWAP by >= stretch, %B >= 0.90, RSI >= 65
                optional: MACD hist declining
  LONG  (calls): mirror below VWAP / lower BB / RSI <= 35

Exit (underlying proxy for options scalp):
  - first touch of VWAP, OR
  - +target MFE (default 0.20%), OR
  - time stop 25m (5 bars)

Usage:
  python3 mr_vwap_scalp.py
  python3 mr_vwap_scalp.py --scan
"""
import argparse
import os
import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.00015
HOLD_BARS = 5          # 25 minutes
DEDUP = 6
STRETCH = 0.0020       # 0.20% from VWAP — modest, matches "not even 0.5%"
TARGET = 0.0020        # 0.20% favorable underlying move
BB_HI, BB_LO = 0.90, 0.10
RSI_HI, RSI_LO = 65, 35


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def macd_hist(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line - sig


def load_5m():
    for name in ("SPY_5m_full.csv", "SPY_5m_yf.csv"):
        path = os.path.join(OUT, name)
        if os.path.exists(path):
            df = pd.read_csv(path)
            col = "timestamps" if "timestamps" in df.columns else (
                "datetime" if "datetime" in df.columns else "ts")
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
        print(f"Fetched yfinance: {len(df)} bars")

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
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["sma9"] = df["close"].rolling(9).mean()
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    df["pctb"] = (df["close"] - df["bb_lo"]) / (4 * sd)
    df["rsi14"] = wilder_rsi(df["close"])
    df["macdh"] = macd_hist(df["close"])
    df["macdh_prev"] = df["macdh"].shift(1)
    return df


def simulate(df, use_macd=False, stretch=STRETCH, target=TARGET,
             hold=HOLD_BARS, exit_vwap=True):
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    dist, pctb, rsi = df["vwap_dist"].values, df["pctb"].values, df["rsi14"].values
    vwap, macdh, macdh_p = df["vwap"].values, df["macdh"].values, df["macdh_prev"].values
    ts, day = df["ts"].values, df["day"].values
    n = len(df)
    trades, last = [], -999

    for i in range(30, n - hold - 1):
        if i - last < DEDUP:
            continue
        d = dist[i]
        short = d >= stretch and pctb[i] >= BB_HI and rsi[i] >= RSI_HI
        long_ = d <= -stretch and pctb[i] <= BB_LO and rsi[i] <= RSI_LO
        if use_macd:
            if short and not (macdh[i] < macdh_p[i]):  # hist shrinking
                short = False
            if long_ and not (macdh[i] > macdh_p[i]):
                long_ = False
        if not (short or long_):
            continue

        side = 1 if short else -1  # 1 = fade/puts
        entry = o[i + 1] * (1 - SLIP if side == 1 else 1 + SLIP)
        exit_px, reason, held = None, "", 0
        for k in range(1, hold + 1):
            j = i + 1 + k - 1
            if j >= n or day[j] != day[i + 1]:
                break
            held = k
            if side == 1:  # short: profit on down
                if h[j] >= entry * (1 + 0.004):  # soft adverse
                    pass
                if l[j] <= entry * (1 - target):
                    exit_px, reason = entry * (1 - target), "target"
                    break
                if exit_vwap and l[j] <= vwap[j]:
                    exit_px, reason = min(entry, vwap[j]), "vwap"
                    # fill at vwap touch approx
                    exit_px = vwap[j]
                    break
            else:
                if h[j] >= entry * (1 + target):
                    exit_px, reason = entry * (1 + target), "target"
                    break
                if exit_vwap and h[j] >= vwap[j]:
                    exit_px, reason = vwap[j], "vwap"
                    break
        if exit_px is None:
            j = min(i + hold, n - 1)
            exit_px, reason, held = c[j], "time", hold

        if side == 1:
            ret = entry / exit_px - 1
            mfe = entry / l[i + 1:i + 1 + held].min() - 1
        else:
            ret = exit_px / entry - 1
            mfe = h[i + 1:i + 1 + held].max() / entry - 1

        trades.append(dict(
            ts=ts[i], day=day[i], side="put" if side == 1 else "call",
            stretch=abs(d), rsi=rsi[i], pctb=pctb[i],
            entry=entry, exit=exit_px, ret=ret, mfe=mfe,
            held_bars=held, reason=reason,
        ))
        last = i
    return pd.DataFrame(trades)


def report(tr, label, n_days):
    if not len(tr):
        print(f"\n{label}: no trades")
        return
    print(f"\n{label}")
    print(f"  n={len(tr)} ({len(tr)/n_days:.2f}/day)  "
          f"puts={(tr['side']=='put').sum()}  calls={(tr['side']=='call').sum()}")
    print(f"  WR={(tr['ret']>0).mean():.0%}  avg={tr['ret'].mean():+.3%}  "
          f"med={tr['ret'].median():+.3%}")
    print(f"  MFE med={tr['mfe'].median():+.3%}  "
          f"P(MFE>=0.15%)={(tr['mfe']>=0.0015).mean():.0%}  "
          f"P(MFE>=0.20%)={(tr['mfe']>=0.002).mean():.0%}  "
          f"P(MFE>=0.30%)={(tr['mfe']>=0.003).mean():.0%}")
    print(f"  exits: " + ", ".join(
        f"{k}={(tr['reason']==k).mean():.0%}" for k in ["target", "vwap", "time"]))
    print(f"  avg hold={tr['held_bars'].mean()*5:.0f}m  "
          f"med stretch at signal={tr['stretch'].median():.2%}")
    for side in ["put", "call"]:
        s = tr[tr["side"] == side]
        if len(s):
            print(f"    {side}: n={len(s)} WR={(s['ret']>0).mean():.0%} "
                  f"avg={s['ret'].mean():+.3%} MFE med={s['mfe'].median():+.3%}")


def vwap_touch_base_rate(df):
    """After any stretch >= X, how often does price touch VWAP within 25m / rest of day?"""
    print("\n" + "=" * 72)
    print("BASE RATE: after VWAP stretch, how often do we get back toward VWAP?")
    print("=" * 72)
    dist = df["vwap_dist"].values
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    vwap, day = df["vwap"].values, df["day"].values
    n = len(df)
    for thr in [0.0015, 0.0020, 0.0030, 0.0040]:
        hits_25, hits_eod, n_ep = 0, 0, 0
        last = -999
        for i in range(20, n - 1):
            if i - last < DEDUP:
                continue
            d = dist[i]
            if abs(d) < thr:
                continue
            n_ep += 1
            last = i
            side = 1 if d > 0 else -1
            touched_25 = False
            for k in range(1, HOLD_BARS + 1):
                j = i + k
                if j >= n or day[j] != day[i]:
                    break
                if side == 1 and l[j] <= vwap[j]:
                    touched_25 = True
                    break
                if side == -1 and h[j] >= vwap[j]:
                    touched_25 = True
                    break
            # rest of day
            touched_eod = touched_25
            if not touched_eod:
                for j in range(i + 1, n):
                    if day[j] != day[i]:
                        break
                    if side == 1 and l[j] <= vwap[j]:
                        touched_eod = True
                        break
                    if side == -1 and h[j] >= vwap[j]:
                        touched_eod = True
                        break
            hits_25 += touched_25
            hits_eod += touched_eod
        print(f"  stretch>={thr:.2%}: n={n_ep}  "
              f"touch VWAP in 25m={hits_25/max(n_ep,1):.0%}  "
              f"by EOD={hits_eod/max(n_ep,1):.0%}")


def scan(df):
    print("\n" + "=" * 72)
    print("LATEST BARS (EMA9/SMA9/BB/VWAP/RSI/MACD)")
    print("=" * 72)
    for _, r in df.tail(10).iterrows():
        flags = []
        d = r["vwap_dist"]
        if d >= STRETCH:
            flags.append(f"aboveVWAP {d:+.2%}")
        if d <= -STRETCH:
            flags.append(f"belowVWAP {d:+.2%}")
        if r["pctb"] >= BB_HI:
            flags.append("upperBB")
        if r["pctb"] <= BB_LO:
            flags.append("lowerBB")
        if r["rsi14"] >= RSI_HI:
            flags.append(f"RSI{r['rsi14']:.0f}")
        if r["rsi14"] <= RSI_LO:
            flags.append(f"RSI{r['rsi14']:.0f}")
        if r["macdh"] < r["macdh_prev"]:
            flags.append("MACD↓")
        if r["macdh"] > r["macdh_prev"]:
            flags.append("MACD↑")
        put = d >= STRETCH and r["pctb"] >= BB_HI and r["rsi14"] >= RSI_HI
        call = d <= -STRETCH and r["pctb"] <= BB_LO and r["rsi14"] <= RSI_LO
        mark = " *** BUY PUTS" if put else (" *** BUY CALLS" if call else "")
        print(f"  {r['ts'].strftime('%m-%d %H:%M')}  c={r['close']:.2f}  "
              f"vwap={r['vwap']:.2f} ({d:+.2%})  %B={r['pctb']:.2f}  "
              f"RSI={r['rsi14']:.0f}  "
              f"{'|'.join(flags) if flags else '-'}{mark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scan", action="store_true")
    args = p.parse_args()
    df = prep(load_5m())
    n_days = df["day"].nunique()
    print(f"RTH {len(df)} bars | {n_days} days | "
          f"{df['ts'].iloc[0].date()} -> {df['ts'].iloc[-1].date()}")
    print(f"Indicators: EMA9, SMA9, BB(20,2), VWAP, RSI14, MACD hist")
    print(f"Stretch>={STRETCH:.2%}  target={TARGET:.2%}  hold={HOLD_BARS*5}m")

    if args.scan:
        scan(df)
        return

    vwap_touch_base_rate(df)

    print("\n" + "=" * 72)
    print("SIGNAL BACKTEST (underlying proxy for options entry timing)")
    print("=" * 72)
    report(simulate(df, use_macd=False),
           "A. VWAP+BB+RSI (no MACD filter)", n_days)
    report(simulate(df, use_macd=True),
           "B. VWAP+BB+RSI + MACD hist turning", n_days)
    report(simulate(df, use_macd=False, stretch=0.0015, target=0.0015),
           "C. Looser stretch 0.15% / target 0.15%", n_days)
    report(simulate(df, use_macd=True, stretch=0.0025, target=0.0025),
           "D. Stricter 0.25% + MACD turn", n_days)

    # save primary
    tr = simulate(df, use_macd=True)
    tr.to_csv(os.path.join(OUT, "mr_vwap_scalp_trades.csv"), index=False)
    scan(df)
    print(f"\nWrote trades -> {OUT}/mr_vwap_scalp_trades.csv")
    print("Note: returns are UNDERLYING. Options (esp. short-dated) magnify these moves.")


if __name__ == "__main__":
    main()
