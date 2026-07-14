#!/usr/bin/env python3
"""Systematic signal combination scan — SPY mean-reversion fades.

Phase 1 — BASE (alone, each TF):
  bb_up / bb_dn     close outside Bollinger(20,2)
  rsi80 / rsi20     RSI14 >= 80 / <= 20

Phase 2 — ADD filters on top of a chosen base (same or higher TF):
  vwap_stretch      |close/vwap-1| >= 0.20% (intraday only)
  rsi65 / rsi35     softer RSI confirm
  pctb_hi / pctb_lo %B >= 0.95 / <= 0.05
  macd_fade         MACD hist declining (shorts) / rising (longs)
  sma9_break        first close under/over SMA9
  htf_bb / htf_rsi  higher-TF BB or RSI extreme confirms

Exit (underlying proxy, consistent across TFs):
  next-bar open entry, hold HOLD_BARS on that TF, slip 1.5bp/side.
  Also report MFE within hold and hit-rate of +0.15% / +0.25%.

Usage:
  python3 signal_combo_scan.py              # phase 1 + phase 2 on best bases
  python3 signal_combo_scan.py --phase 1
  python3 signal_combo_scan.py --phase 2 --base bb_up --tf 5m
"""
from __future__ import annotations

import argparse
import itertools
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.00015
HOLD_BARS = 5
DEDUP = 5
TARGET_A, TARGET_B = 0.0015, 0.0025

TFS = {
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "1d": "1d",
}


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
    return df


def load_daily():
    path = os.path.join(OUT, "SPY_daily.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        col = next(c for c in ("timestamps", "datetime", "ts", "date", "Date") if c in df.columns)
        df["ts"] = pd.to_datetime(df[col])
        if df["ts"].dt.tz is None:
            df["ts"] = df["ts"].dt.tz_localize(NY)
        else:
            df["ts"] = df["ts"].dt.tz_convert(NY)
        # keep last ~3y for daily signal density
        df = df.sort_values("ts").tail(780).reset_index(drop=True)
    else:
        import yfinance as yf
        raw = yf.download("SPY", interval="1d", period="3y",
                          auto_adjust=True, progress=False).reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                           for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        ts_col = "date" if "date" in raw.columns else raw.columns[0]
        df = raw.rename(columns={ts_col: "ts"})
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(NY)
        df.to_csv(path, index=False)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # align tz/resolution with intraday frames
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    print(f"Loaded daily: {len(df)} bars ({df['ts'].iloc[0].date()} -> {df['ts'].iloc[-1].date()})")
    return df


def resample(df5, rule):
    if rule == "1d":
        return None  # use dedicated daily
    x = df5.set_index("ts")
    ohlc = x.resample(rule, label="right", closed="right").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["close"]).reset_index()
    # drop incomplete RTH buckets with tiny volume
    ohlc = ohlc[ohlc["volume"] > 0].reset_index(drop=True)
    return ohlc


def prep(df, intraday=True):
    df = df.copy()
    df["day"] = df["ts"].dt.date
    mid = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    df["bb_mid"], df["bb_up"], df["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    df["pctb"] = (df["close"] - df["bb_lo"]) / (4 * sd.replace(0, np.nan))
    df["rsi"] = wilder_rsi(df["close"])
    df["macdh"] = macd_hist(df["close"])
    df["sma9"] = df["close"].rolling(9).mean()
    if intraday:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_pv = (tp * df["volume"]).groupby(df["day"]).cumsum()
        cum_v = df["volume"].groupby(df["day"]).cumsum().replace(0, np.nan)
        df["vwap"] = cum_pv / cum_v
        df["vwap_dist"] = df["close"] / df["vwap"] - 1.0
    else:
        df["vwap"] = np.nan
        df["vwap_dist"] = np.nan
    return df


def base_mask(df, name):
    """Return boolean Series for base extreme."""
    if name == "bb_up":
        return df["close"] >= df["bb_up"]
    if name == "bb_dn":
        return df["close"] <= df["bb_lo"]
    if name == "rsi80":
        return df["rsi"] >= 80
    if name == "rsi20":
        return df["rsi"] <= 20
    raise KeyError(name)


def side_of(base):
    """Fade direction: short on overbought, long on oversold."""
    return "short" if base in ("bb_up", "rsi80") else "long"


def filter_mask(df, name, side, htf=None):
    """Optional confluence filter. htf is a prepared higher-TF frame aligned via merge_asof."""
    if name == "none":
        return pd.Series(True, index=df.index)
    if name == "vwap_stretch":
        if df["vwap_dist"].isna().all():
            return pd.Series(False, index=df.index)
        return df["vwap_dist"] >= 0.0020 if side == "short" else df["vwap_dist"] <= -0.0020
    if name == "rsi65":
        return df["rsi"] >= 65 if side == "short" else df["rsi"] <= 35
    if name == "pctb_edge":
        return df["pctb"] >= 0.95 if side == "short" else df["pctb"] <= 0.05
    if name == "macd_fade":
        d = df["macdh"].diff()
        return d < 0 if side == "short" else d > 0
    if name == "sma9_break":
        # same-bar SMA9 break (rarely coincides with BB extreme)
        above = df["close"] > df["sma9"]
        below = df["close"] < df["sma9"]
        if side == "short":
            return below & above.shift(1).fillna(False)
        return above & below.shift(1).fillna(False)
    if name == "htf_bb":
        if htf is None:
            return pd.Series(False, index=df.index)
        col = "bb_up_htf" if side == "short" else "bb_dn_htf"
        return htf[col].reindex(df.index).fillna(False).astype(bool)
    if name == "htf_rsi":
        if htf is None:
            return pd.Series(False, index=df.index)
        col = "rsi80_htf" if side == "short" else "rsi20_htf"
        return htf[col].reindex(df.index).fillna(False).astype(bool)
    raise KeyError(name)


def delayed_sma9_mask(df, base_first, side, look=8):
    """After a base extreme, wait up to `look` bars for first SMA9 break (more realistic trigger)."""
    above = (df["close"] > df["sma9"]).values
    below = (df["close"] < df["sma9"]).values
    base = base_first.fillna(False).values
    n = len(df)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        if not base[i]:
            continue
        for k in range(1, look + 1):
            j = i + k
            if j >= n:
                break
            if side == "short":
                if below[j] and above[j - 1]:
                    out[j] = True
                    break
            else:
                if above[j] and below[j - 1]:
                    out[j] = True
                    break
    return pd.Series(out, index=df.index)


def align_htf(df, htf_df):
    """As-of align higher TF flags onto lower TF bars (no lookahead: prior completed HTF bar)."""
    # merge on UTC-ns ints to avoid tz/resolution mismatches across TFs
    left = pd.DataFrame({
        "ts_ns": pd.to_datetime(df["ts"], utc=True).astype("int64").values,
    }, index=df.index)
    h = pd.DataFrame({
        "ts_ns": pd.to_datetime(htf_df["ts"], utc=True).astype("int64").values,
        "bb_up_htf": (htf_df["close"].values >= htf_df["bb_up"].values),
        "bb_dn_htf": (htf_df["close"].values <= htf_df["bb_lo"].values),
        "rsi80_htf": (htf_df["rsi"].values >= 80),
        "rsi20_htf": (htf_df["rsi"].values <= 20),
    }).sort_values("ts_ns")
    for c in ["bb_up_htf", "bb_dn_htf", "rsi80_htf", "rsi20_htf"]:
        h[c] = h[c].shift(1)
    out = pd.merge_asof(
        left.sort_values("ts_ns"),
        h.dropna(subset=["bb_up_htf"]),
        on="ts_ns",
        direction="backward",
    )
    return out.reindex(df.index)


def backtest(df, entry_mask, side, label="", hold=HOLD_BARS, dedup=DEDUP):
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    day = df["day"].values
    mask = entry_mask.fillna(False).values
    n = len(df)
    trades, last = [], -10**9

    for i in range(25, n - hold - 1):
        if not mask[i]:
            continue
        if i - last < dedup:
            continue
        # entry next open
        entry = o[i + 1] * (1 + SLIP if side == "long" else 1 - SLIP)
        end = min(i + 1 + hold, n)
        # same-session preference for intraday
        path_h = h[i + 1:end]
        path_l = l[i + 1:end]
        path_c = c[i + 1:end]
        if len(path_c) == 0:
            continue
        exit_px = path_c[-1]
        if side == "short":
            # sell high, cover lower -> profit when price falls
            ret = entry / exit_px - 1.0
            mfe = entry / path_l.min() - 1.0 if len(path_l) else 0.0
            mae = entry / path_h.max() - 1.0 if len(path_h) else 0.0
        else:
            ret = exit_px / entry - 1.0
            mfe = path_h.max() / entry - 1.0 if len(path_h) else 0.0
            mae = path_l.min() / entry - 1.0 if len(path_l) else 0.0
        trades.append(dict(
            ts=df["ts"].iloc[i], day=day[i], side=side,
            entry=entry, exit=exit_px, ret=ret, mfe=mfe, mae=mae,
            held=len(path_c),
        ))
        last = i

    tr = pd.DataFrame(trades)
    n_days = max(int(df["day"].nunique()), 1)
    row = dict(
        label=label, n=len(tr), per_day=len(tr) / n_days,
        wr=float((tr["ret"] > 0).mean()) if len(tr) else np.nan,
        avg=float(tr["ret"].mean()) if len(tr) else np.nan,
        med=float(tr["ret"].median()) if len(tr) else np.nan,
        sum=float(tr["ret"].sum()) if len(tr) else 0.0,
        mfe_med=float(tr["mfe"].median()) if len(tr) else np.nan,
        hit15=float((tr["mfe"] >= TARGET_A).mean()) if len(tr) else np.nan,
        hit25=float((tr["mfe"] >= TARGET_B).mean()) if len(tr) else np.nan,
    )
    return tr, row


def fmt_row(r):
    if r["n"] == 0 or np.isnan(r["wr"]):
        return f"  {r['label']:<42} n=0"
    return (
        f"  {r['label']:<42} n={r['n']:>4} ({r['per_day']:.2f}/d)  "
        f"WR={r['wr']:.0%}  avg={r['avg']:+.3%}  med={r['med']:+.3%}  "
        f"sum={r['sum']:+.2%}  MFE med={r['mfe_med']:+.3%}  "
        f"P≥0.15%={r['hit15']:.0%}  P≥0.25%={r['hit25']:.0%}"
    )


def build_frames(df5, daily):
    frames = {}
    for tf, rule in TFS.items():
        if tf == "1d":
            frames[tf] = prep(daily, intraday=False)
        else:
            raw = df5 if tf == "5m" else resample(df5, rule)
            frames[tf] = prep(raw, intraday=True)
        print(f"  {tf}: {len(frames[tf])} bars | {frames[tf]['day'].nunique()} days")
    return frames


def phase1(frames):
    print("\n" + "=" * 88)
    print("PHASE 1 — BASE SIGNALS (fade extremes, hold 5 bars on that TF)")
    print("=" * 88)
    bases = ["bb_up", "bb_dn", "rsi80", "rsi20"]
    rows = []
    for tf, df in frames.items():
        print(f"\n-- {tf} --")
        for b in bases:
            side = side_of(b)
            mask = base_mask(df, b)
            # trigger on first bar of extreme (not every bar while extreme)
            first = mask & ~mask.shift(1).fillna(False)
            _, row = backtest(df, first, side, label=f"{tf}|{b}|fade")
            print(fmt_row(row))
            rows.append({**row, "tf": tf, "base": b, "side": side})
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "signal_combo_phase1.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")

    # ranking: require n>=8, sort by avg then wr
    ranked = out[out["n"] >= 8].sort_values(["avg", "wr"], ascending=False)
    print("\nTOP BASE (n>=8) by avg return:")
    for _, r in ranked.head(8).iterrows():
        print(fmt_row(r))
    print("\nBOTTOM BASE (n>=8):")
    for _, r in ranked.tail(5).iterrows():
        print(fmt_row(r))
    return out


def phase2(frames, base_name, tf, max_combo=2):
    """Stack 1..max_combo filters onto a base signal."""
    print("\n" + "=" * 88)
    print(f"PHASE 2 — ADD FILTERS on {tf}|{base_name} (combinations up to {max_combo})")
    print("=" * 88)
    df = frames[tf]
    side = side_of(base_name)
    base = base_mask(df, base_name)
    first = base & ~base.shift(1).fillna(False)

    # pick an HTF one step up when possible
    htf_map = {"5m": "15m", "15m": "30m", "30m": "1h", "1h": "1d", "1d": None}
    htf_name = htf_map.get(tf)
    htf_aligned = None
    if htf_name and htf_name in frames:
        htf_aligned = align_htf(df, frames[htf_name])

    filters = ["vwap_stretch", "rsi65", "pctb_edge", "macd_fade", "sma9_break"]
    if htf_aligned is not None:
        filters += ["htf_bb", "htf_rsi"]
    # drop intraday-only filters on daily
    if tf == "1d":
        filters = [f for f in filters if f not in ("vwap_stretch",)]

    rows = []
    # baseline
    _, row = backtest(df, first, side, label=f"{tf}|{base_name}|alone")
    print(fmt_row(row))
    rows.append({**row, "tf": tf, "base": base_name, "filters": "alone"})

    # delayed SMA9 trigger after base extreme (structurally valid; same-bar is usually empty)
    delayed = delayed_sma9_mask(df, first, side, look=8)
    _, row = backtest(df, delayed, side, label=f"{tf}|{base_name}>>sma9_break")
    print(fmt_row(row))
    rows.append({**row, "tf": tf, "base": base_name, "filters": ">>sma9_break"})
    delayed_vwap = delayed & filter_mask(df, "vwap_stretch", side, htf_aligned)
    # vwap at the *base* bar is more meaningful; approximate via rolling any of last 8
    # keep simple: require stretch on the break bar itself
    _, row = backtest(df, delayed_vwap, side, label=f"{tf}|{base_name}>>sma9+vwap")
    print(fmt_row(row))
    rows.append({**row, "tf": tf, "base": base_name, "filters": ">>sma9+vwap"})

    # singles
    print("\n-- single filters --")
    for f in filters:
        m = first & filter_mask(df, f, side, htf_aligned)
        _, row = backtest(df, m, side, label=f"{tf}|{base_name}+{f}")
        print(fmt_row(row))
        rows.append({**row, "tf": tf, "base": base_name, "filters": f})

    # pairs
    if max_combo >= 2:
        print("\n-- filter pairs --")
        for a, b in itertools.combinations(filters, 2):
            m = first & filter_mask(df, a, side, htf_aligned) & filter_mask(df, b, side, htf_aligned)
            _, row = backtest(df, m, side, label=f"{tf}|{base_name}+{a}+{b}")
            if row["n"] >= 5:
                print(fmt_row(row))
            rows.append({**row, "tf": tf, "base": base_name, "filters": f"{a}+{b}"})

    out = pd.DataFrame(rows)
    path = os.path.join(OUT, f"signal_combo_phase2_{tf}_{base_name}.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")

    usable = out[out["n"] >= 8].copy()
    if len(usable):
        usable["score"] = usable["avg"] * np.sqrt(usable["n"].clip(lower=1))
        usable = usable.sort_values("score", ascending=False)
        print("\nBEST COMBOS (n>=8, scored avg*sqrt(n)):")
        for _, r in usable.head(10).iterrows():
            print(fmt_row(r))
    return out


def auto_phase2(frames, phase1_df, top_k=4):
    """Run phase2 on strongest phase1 bases, preferring intraday (options-scalp relevant)."""
    ranked = phase1_df[phase1_df["n"] >= 8].sort_values("avg", ascending=False)
    # Prefer intraday first for the scalp workflow; still include best daily if strong.
    intra = ranked[ranked["tf"] != "1d"]
    picks = []
    for pool in (intra, ranked):
        for _, r in pool.iterrows():
            key = (r["tf"], r["base"])
            if key not in picks:
                picks.append(key)
            if len(picks) >= top_k:
                break
        if len(picks) >= top_k:
            break
    # Always include best short + long intraday bases when available
    for side in ("short", "long"):
        sub = intra[intra["side"] == side]
        if len(sub):
            key = (sub.iloc[0]["tf"], sub.iloc[0]["base"])
            if key not in picks:
                picks.append(key)
    # Force the core BB bases on 5m/15m — primary research lane
    for tf, base in [("5m", "bb_dn"), ("5m", "bb_up"), ("15m", "bb_dn"), ("15m", "bb_up")]:
        if (tf, base) not in picks:
            picks.append((tf, base))

    print(f"\nPhase-2 targets: {picks}")
    all_rows = []
    for tf, base in picks:
        all_rows.append(phase2(frames, base, tf, max_combo=2))
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "2", "all"], default="all")
    ap.add_argument("--base", default=None, help="e.g. bb_up")
    ap.add_argument("--tf", default=None, help="e.g. 5m")
    ap.add_argument("--top", type=int, default=4)
    args = ap.parse_args()

    print("Building frames…")
    df5 = load_5m()
    daily = load_daily()
    frames = build_frames(df5, daily)
    print(f"Intraday window: {df5['ts'].iloc[0].date()} -> {df5['ts'].iloc[-1].date()}")
    print(f"Hold={HOLD_BARS} bars/TF | slip={SLIP:.2%}/side | fade MR exits at bar close")

    p1 = None
    if args.phase in ("1", "all"):
        p1 = phase1(frames)

    if args.phase in ("2", "all"):
        if args.base and args.tf:
            phase2(frames, args.base, args.tf, max_combo=2)
        else:
            if p1 is None:
                path = os.path.join(OUT, "signal_combo_phase1.csv")
                p1 = pd.read_csv(path)
            auto_phase2(frames, p1, top_k=args.top)


if __name__ == "__main__":
    main()
