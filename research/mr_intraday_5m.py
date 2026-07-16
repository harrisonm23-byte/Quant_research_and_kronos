#!/usr/bin/env python3
"""SPY 5m intraday mean-reversion study + confluence scanner.

Measures:
  1. How often "large" VWAP stretches occur and snap back (base rate)
  2. Daily precursors of high-MR days
  3. VWAP + BB + RSI confluence signal with 20-25 min (4-5 bar) holds

Data: prefers research/SPY_5m_full.csv; else fetches ~60d via yfinance.

Usage:
  python3 mr_intraday_5m.py
  python3 mr_intraday_5m.py --fetch   # force yfinance refresh
"""
import argparse
import math
import os
import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.0002

# stretch / snap definitions
STRETCH_PCT = 0.0035      # |close-VWAP|/VWAP >= 0.35%
SNAP_FRAC = 0.50          # reclaim >= 50% of stretch within horizon
HOLD_BARS = 5             # 5 × 5m = 25 minutes
DEDUP_BARS = 6            # ignore new arms within 30m of prior signal


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def load_5m(force_fetch=False):
    path = os.path.join(OUT, "SPY_5m_full.csv")
    yf_path = os.path.join(OUT, "SPY_5m_yf.csv")
    if not force_fetch and os.path.exists(path):
        df = pd.read_csv(path)
        col = "timestamps" if "timestamps" in df.columns else "datetime"
        df["ts"] = pd.to_datetime(df[col], utc=True).dt.tz_convert(NY)
        print(f"Loaded {path}: {len(df)} bars")
    else:
        try:
            import yfinance as yf
        except ImportError:
            sys.exit("Need yfinance or SPY_5m_full.csv")
        raw = yf.download("SPY", interval="5m", period="60d",
                          auto_adjust=True, progress=False)
        raw = raw.reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                           for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        ts_col = "datetime" if "datetime" in raw.columns else raw.columns[0]
        df = raw.rename(columns={ts_col: "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        df.to_csv(yf_path, index=False)
        print(f"Fetched yfinance 5m: {len(df)} bars -> {yf_path}")

    keep = (df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))
    df = df.loc[keep].sort_values("ts").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["day"] = df["ts"].dt.date
    return df


def prep(df):
    # session VWAP
    pv = df["close"] * df["volume"]  # approx if no typical price
    # use typical price for VWAP when possible
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    cum_pv = pv.groupby(df["day"]).cumsum()
    cum_v = df["volume"].groupby(df["day"]).cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_v
    df["vwap_dist"] = df["close"] / df["vwap"] - 1.0  # + = above VWAP

    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    df["bb_mid"] = mid
    df["bb_up"] = mid + 2 * sd
    df["bb_lo"] = mid - 2 * sd
    df["pctb"] = (df["close"] - df["bb_lo"]) / (4 * sd)

    df["rsi14"] = wilder_rsi(df["close"], 14)
    df["ret1"] = df["close"].pct_change()
    df["volx"] = df["volume"] / df["volume"].rolling(20).mean()
    # bar index within day
    df["bar_i"] = df.groupby("day").cumcount()
    return df


# ---------- 1. base-rate: how often do large stretches / snaps happen? ----------
def base_rate(df):
    print("\n" + "=" * 72)
    print("1. HOW OFTEN DO LARGE MEAN-REVERSIONS OCCUR? (SPY 5m)")
    print("=" * 72)
    print(f"Stretch = |close-VWAP|/VWAP >= {STRETCH_PCT:.2%}")
    print(f"Snap    = reclaim >= {SNAP_FRAC:.0%} of stretch within {HOLD_BARS} bars (25m)")

    days = sorted(df["day"].unique())
    n_days = len(days)
    stretch_events = []  # per event
    day_stats = []

    for day, g in df.groupby("day"):
        g = g.reset_index(drop=True)
        o, h, l, c = g["open"].values, g["high"].values, g["low"].values, g["close"].values
        dist = g["vwap_dist"].values
        n = len(g)
        # peak stretch of day (max |dist|)
        peak_i = int(np.nanargmax(np.abs(dist)))
        peak = dist[peak_i]
        # did it snap within HOLD_BARS after peak?
        j = min(peak_i + HOLD_BARS, n - 1)
        if peak > 0:
            # above VWAP — snap = price falls toward VWAP
            snap_amt = (c[peak_i] - l[peak_i:j + 1].min()) / c[peak_i]
            toward = peak > 0 and (c[peak_i] - l[peak_i:j + 1].min()) >= SNAP_FRAC * abs(peak) * c[peak_i]
            # simpler: min close in window vs peak close
            mfe_against = (c[peak_i] - l[peak_i:j + 1].min()) / c[peak_i]
            snapped = mfe_against >= SNAP_FRAC * abs(peak)
        else:
            mfe_against = (h[peak_i:j + 1].max() - c[peak_i]) / c[peak_i]
            snapped = mfe_against >= SNAP_FRAC * abs(peak)

        # count distinct stretch episodes (deduped)
        armed = False
        episodes = 0
        snaps = 0
        for i in range(n - HOLD_BARS):
            d = dist[i]
            if abs(d) < STRETCH_PCT:
                armed = False
                continue
            if armed:
                continue
            # new episode
            armed = True
            episodes += 1
            # side: +1 fade (short), -1 bounce (long)
            side = 1 if d > 0 else -1
            entry = c[i]  # conservative: next-bar open better — use next open
            if i + 1 >= n:
                continue
            entry = o[i + 1] * (1 + SLIP * side)  # short: sell lower? slip against
            # for short entry: fill at open*(1-slip); long: open*(1+slip)
            entry = o[i + 1] * (1 - SLIP) if side == 1 else o[i + 1] * (1 + SLIP)
            win = h[i + 1:i + 1 + HOLD_BARS] if side == -1 else l[i + 1:i + 1 + HOLD_BARS]
            if side == 1:  # short: profit if price drops
                mfe = entry / win.min() - 1 if len(win) else 0
                mae = win.max() / entry - 1 if len(win) else 0
                end = (entry / c[min(i + HOLD_BARS, n - 1)] - 1)
            else:
                mfe = win.max() / entry - 1 if len(win) else 0
                mae = entry / win.min() - 1 if len(win) else 0
                end = c[min(i + HOLD_BARS, n - 1)] / entry - 1
            snapped_ep = mfe >= SNAP_FRAC * abs(d)
            if snapped_ep:
                snaps += 1
            stretch_events.append(dict(
                day=day, side=side, stretch=abs(d), mfe=mfe, mae=mae, end=end,
                snapped=snapped_ep, bar=g["ts"].iloc[i],
            ))
            # keep armed until stretch collapses
            # (simple: stay armed while still stretched)

        day_stats.append(dict(
            day=day, peak_stretch=abs(peak), peak_side=np.sign(peak),
            peak_snapped=snapped, episodes=episodes, snaps=snaps,
            day_range=(g["high"].max() / g["low"].min() - 1),
        ))

    ds = pd.DataFrame(day_stats)
    ev = pd.DataFrame(stretch_events)
    print(f"\nDays in sample: {n_days}")
    print(f"Days with peak |VWAP stretch| >= {STRETCH_PCT:.2%}: "
          f"{(ds['peak_stretch'] >= STRETCH_PCT).mean():.0%} "
          f"({(ds['peak_stretch'] >= STRETCH_PCT).sum()} days)")
    print(f"Days with peak stretch >= 0.50%: {(ds['peak_stretch'] >= 0.005).mean():.0%}")
    print(f"Days with peak stretch >= 0.75%: {(ds['peak_stretch'] >= 0.0075).mean():.0%}")
    print(f"Median peak daily stretch: {ds['peak_stretch'].median():.2%}")
    print(f"Mean peak daily stretch:   {ds['peak_stretch'].mean():.2%}")

    print(f"\nStretch episodes (|dist|>={STRETCH_PCT:.2%}, deduped): {len(ev)}")
    print(f"  per day: {len(ev)/n_days:.2f}")
    print(f"  snap within 25m (>=50% reclaim): {ev['snapped'].mean():.0%}")
    print(f"  avg MFE (favorable): {ev['mfe'].mean():.3%}  med={ev['mfe'].median():.3%}")
    print(f"  avg 25m P&L (dir):   {ev['end'].mean():.3%}  WR={(ev['end']>0).mean():.0%}")

    # frequency of "at least one good snap day"
    good = ds["snaps"] >= 1
    print(f"\nDays with >=1 snapped episode: {good.mean():.0%} ({good.sum()}/{n_days})")
    print(f"Days with >=2 snapped episodes: {(ds['snaps']>=2).mean():.0%}")

    return ds, ev


# ---------- 2. precursors of high-MR days ----------
def precursors(df, ds):
    print("\n" + "=" * 72)
    print("2. WHAT PRECEDES HIGH MEAN-REVERSION DAYS?")
    print("=" * 72)

    daily = df.groupby("day").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
        first_vwap=("vwap", "first"),
    ).reset_index()
    daily["ret"] = daily["close"] / daily["open"] - 1
    daily["range"] = daily["high"] / daily["low"] - 1
    daily["gap"] = daily["open"] / daily["close"].shift(1) - 1
    daily["prev_ret"] = daily["ret"].shift(1)
    daily["prev_range"] = daily["range"].shift(1)
    daily["volx"] = daily["volume"] / daily["volume"].rolling(10).mean()

    # overnight / prior close context from prior day close vs open stretch
    merged = daily.merge(ds[["day", "peak_stretch", "snaps"]], on="day")
    # high-MR day = peak stretch >= 0.5% OR snaps >= 1
    merged["high_mr"] = (merged["peak_stretch"] >= 0.005) | (merged["snaps"] >= 1)

    # precursors known at OPEN (or prior close)
    feats = [
        ("|gap| >= 0.30%", lambda r: abs(r["gap"]) >= 0.003 if not math.isnan(r["gap"]) else False),
        ("gap up >= 0.20%", lambda r: r["gap"] >= 0.002 if not math.isnan(r["gap"]) else False),
        ("gap down <= -0.20%", lambda r: r["gap"] <= -0.002 if not math.isnan(r["gap"]) else False),
        ("prior day range >= 1%", lambda r: r["prev_range"] >= 0.01 if not math.isnan(r["prev_range"]) else False),
        ("prior day |ret| >= 0.5%", lambda r: abs(r["prev_ret"]) >= 0.005 if not math.isnan(r["prev_ret"]) else False),
        ("prior day up", lambda r: r["prev_ret"] > 0 if not math.isnan(r["prev_ret"]) else False),
        ("prior day down", lambda r: r["prev_ret"] < 0 if not math.isnan(r["prev_ret"]) else False),
        ("volume >= 1.2x 10d", lambda r: r["volx"] >= 1.2 if not math.isnan(r["volx"]) else False),
    ]

    base = merged["high_mr"].mean()
    print(f"Base rate high-MR day: {base:.0%} (n={len(merged)})")
    print(f"\n{'precursor (known by open)':<28s}{'P(MR|feat)':>11s}{'lift':>7s}{'n':>5s}")
    for name, fn in feats:
        mask = merged.apply(fn, axis=1)
        if mask.sum() < 5:
            continue
        p = merged.loc[mask, "high_mr"].mean()
        print(f"{name:<28s}{p:>11.0%}{p-base:>+7.0%}{int(mask.sum()):>5d}")

    # also: time-of-day when peak stretch prints
    print("\nWhen does the day's peak |VWAP stretch| usually print?")
    peaks = []
    for day, g in df.groupby("day"):
        dist = g["vwap_dist"].values
        i = int(np.nanargmax(np.abs(dist)))
        peaks.append(g["ts"].iloc[i].hour * 60 + g["ts"].iloc[i].minute)
    peaks = np.array(peaks)
    buckets = [(9 * 60 + 30, 10 * 60, "09:30-10:00"),
               (10 * 60, 11 * 60, "10:00-11:00"),
               (11 * 60, 12 * 60, "11:00-12:00"),
               (12 * 60, 13 * 60, "12:00-13:00"),
               (13 * 60, 14 * 60, "13:00-14:00"),
               (14 * 60, 15 * 60, "14:00-15:00"),
               (15 * 60, 16 * 60, "15:00-16:00")]
    for lo, hi, lab in buckets:
        share = ((peaks >= lo) & (peaks < hi)).mean()
        print(f"  {lab}: {share:.0%}")


# ---------- 3. confluence scanner (your rules) ----------
def confluence_backtest(df):
    print("\n" + "=" * 72)
    print("3. CONFLUENCE SIGNAL: VWAP stretch + BB edge + RSI extreme")
    print("   Hold = 25 minutes (5 × 5m bars) | dedupe 30m")
    print("=" * 72)

    # SHORT: above VWAP, upper BB / high RSI
    # LONG:  below VWAP, lower BB / low RSI
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    dist = df["vwap_dist"].values
    pctb = df["pctb"].values
    rsi = df["rsi14"].values
    ts = df["ts"].values
    n = len(df)

    def run(vwap_thr, bb_mode, rsi_lo, rsi_hi, label):
        trades = []
        last_sig = -999
        for i in range(25, n - HOLD_BARS - 1):
            if i - last_sig < DEDUP_BARS:
                continue
            d = dist[i]
            short = (d >= vwap_thr and pctb[i] >= 0.95 and rsi[i] >= rsi_hi)
            long_ = (d <= -vwap_thr and pctb[i] <= 0.05 and rsi[i] <= rsi_lo)
            if not (short or long_):
                continue
            side = 1 if short else -1  # 1=short fade
            # optional: require only 2 of 3 — also report strict 3/3
            entry = o[i + 1] * (1 - SLIP) if side == 1 else o[i + 1] * (1 + SLIP)
            j = min(i + 1 + HOLD_BARS, n - 1)
            if side == 1:
                mfe = entry / l[i + 1:j + 1].min() - 1
                mae = h[i + 1:j + 1].max() / entry - 1
                end = entry / c[j] - 1
            else:
                mfe = h[i + 1:j + 1].max() / entry - 1
                mae = entry / l[i + 1:j + 1].min() - 1
                end = c[j] / entry - 1
            trades.append(dict(side=side, end=end, mfe=mfe, mae=mae,
                               stretch=abs(d), rsi=rsi[i], ts=ts[i]))
            last_sig = i
        tr = pd.DataFrame(trades)
        if not len(tr):
            print(f"\n{label}: no trades")
            return tr
        print(f"\n{label}")
        print(f"  n={len(tr)}  ({len(tr)/df['day'].nunique():.2f}/day)  "
              f"short={(tr['side']==1).sum()}  long={(tr['side']==-1).sum()}")
        print(f"  25m WR={(tr['end']>0).mean():.0%}  avg={tr['end'].mean():+.3%}  "
              f"med={tr['end'].median():+.3%}")
        print(f"  MFE med={tr['mfe'].median():+.3%}  MAE med={tr['mae'].median():+.3%}")
        print(f"  P(MFE>=0.15%)={(tr['mfe']>=0.0015).mean():.0%}  "
              f"P(MFE>=0.25%)={(tr['mfe']>=0.0025).mean():.0%}")
        return tr

    # strict 3/3
    run(0.0025, "edge", 30, 70, "STRICT 3/3: |VWAP|>=0.25% + BB edge + RSI 30/70")
    run(0.0035, "edge", 30, 70, "STRICT 3/3: |VWAP|>=0.35% + BB edge + RSI 30/70")
    run(0.0025, "edge", 35, 65, "STRICT 3/3: |VWAP|>=0.25% + BB edge + RSI 35/65 (looser)")

    # 2-of-3 variants
    print("\n--- 2-of-3 variants (any two of VWAP/BB/RSI) ---")
    trades = []
    last_sig = -999
    vwap_thr, rsi_lo, rsi_hi = 0.0025, 30, 70
    for i in range(25, n - HOLD_BARS - 1):
        if i - last_sig < DEDUP_BARS:
            continue
        d = dist[i]
        # short ingredients
        s_vwap = d >= vwap_thr
        s_bb = pctb[i] >= 0.95
        s_rsi = rsi[i] >= rsi_hi
        l_vwap = d <= -vwap_thr
        l_bb = pctb[i] <= 0.05
        l_rsi = rsi[i] <= rsi_lo
        short = sum([s_vwap, s_bb, s_rsi]) >= 2 and (s_vwap or s_bb)  # need stretch or band
        long_ = sum([l_vwap, l_bb, l_rsi]) >= 2 and (l_vwap or l_bb)
        if not (short or long_):
            continue
        # prefer short if both (rare)
        side = 1 if short else -1
        entry = o[i + 1] * (1 - SLIP) if side == 1 else o[i + 1] * (1 + SLIP)
        j = min(i + 1 + HOLD_BARS, n - 1)
        if side == 1:
            mfe = entry / l[i + 1:j + 1].min() - 1
            end = entry / c[j] - 1
        else:
            mfe = h[i + 1:j + 1].max() / entry - 1
            end = c[j] / entry - 1
        trades.append(dict(side=side, end=end, mfe=mfe, stretch=abs(d)))
        last_sig = i
    tr = pd.DataFrame(trades)
    if len(tr):
        print(f"2-of-3 |VWAP|>=0.25% or BB + RSI band:")
        print(f"  n={len(tr)} ({len(tr)/df['day'].nunique():.2f}/day)  "
              f"WR={(tr['end']>0).mean():.0%} avg={tr['end'].mean():+.3%} "
              f"MFE med={tr['mfe'].median():+.3%}")

    return tr


def scan_latest(df):
    """Print current confluence state on the latest bars (alert helper)."""
    print("\n" + "=" * 72)
    print("4. LATEST BARS — confluence scanner (alert mode)")
    print("=" * 72)
    tail = df.tail(12).copy()
    for _, r in tail.iterrows():
        d = r["vwap_dist"]
        flags = []
        if d >= 0.0035:
            flags.append(f"ABOVE_VWAP {d:+.2%}")
        if d <= -0.0035:
            flags.append(f"BELOW_VWAP {d:+.2%}")
        if r["pctb"] >= 0.95:
            flags.append("UPPER_BB")
        if r["pctb"] <= 0.05:
            flags.append("LOWER_BB")
        if r["rsi14"] >= 70:
            flags.append(f"RSI_OB {r['rsi14']:.0f}")
        if r["rsi14"] <= 30:
            flags.append(f"RSI_OS {r['rsi14']:.0f}")
        short = d >= 0.0035 and r["pctb"] >= 0.95 and r["rsi14"] >= 70
        long_ = d <= -0.0035 and r["pctb"] <= 0.05 and r["rsi14"] <= 30
        mark = ""
        if short:
            mark = " *** SHORT FADE ARMED"
        elif long_:
            mark = " *** LONG BOUNCE ARMED"
        ts = r["ts"].strftime("%Y-%m-%d %H:%M")
        extra = (" | " + ", ".join(flags)) if flags else ""
        print(f"  {ts}  c={r['close']:.2f}  vwap_dist={d:+.2%}  "
              f"%B={r['pctb']:.2f}  RSI={r['rsi14']:.0f}{extra}{mark}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch", action="store_true")
    p.add_argument("--scan", action="store_true", help="Only print latest confluence state")
    args = p.parse_args()
    df = load_5m(force_fetch=args.fetch)
    df = prep(df)
    print(f"RTH bars: {len(df)}  days: {df['day'].nunique()}  "
          f"{df['ts'].iloc[0].date()} -> {df['ts'].iloc[-1].date()}")
    if args.scan:
        scan_latest(df)
        return
    ds, ev = base_rate(df)
    precursors(df, ds)
    confluence_backtest(df)
    scan_latest(df)
    ds.to_csv(os.path.join(OUT, "mr_intraday_day_stats.csv"), index=False)
    ev.to_csv(os.path.join(OUT, "mr_intraday_stretch_events.csv"), index=False)
    print(f"\nWrote day stats + stretch events to {OUT}/")


if __name__ == "__main__":
    main()
