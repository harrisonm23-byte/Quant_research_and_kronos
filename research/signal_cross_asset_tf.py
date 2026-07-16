#!/usr/bin/env python3
"""Cross-asset + multi-TF port of L1/L2/L3 long recipes (and short mirrors).

Symbols: SPY, QQQ, IWM, DIA
TFs: 5m, 15m, 30m, 1h
Recipes: raw bb_dn/up, L1/L2/L3 (+ VIX rising when available on 5m/15m)

Also tests short-side mirrors:
  S1: bb_up + prior_down + rsi65
  S_stretch: bb_up + vwap stretch + rsi65 (15m-style)

Usage:
  python3 signal_cross_asset_tf.py
  python3 signal_cross_asset_tf.py --clock-25m
  python3 signal_cross_asset_tf.py --min-n 10
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
sys.path.insert(0, OUT)

import signal_combo_scan as s
import signal_combo_phase3 as p3
import signal_vix_study as vx

SYMBOLS = ["SPY", "QQQ", "IWM", "DIA"]
TFS = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h"}


def load_symbol_5m(sym):
    path = os.path.join(OUT, f"{sym}_5m_yf.csv")
    if not os.path.exists(path):
        import yfinance as yf
        raw = yf.download(sym, interval="5m", period="60d",
                          auto_adjust=True, progress=False).reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                           for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        ts = "datetime" if "datetime" in raw.columns else raw.columns[0]
        df = raw.rename(columns={ts: "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_csv(path, index=False)
        print(f"Fetched {sym}: {len(df)}")
    else:
        df = pd.read_csv(path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        print(f"Loaded {path}: {len(df)}")
    keep = (df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))
    return df.loc[keep].sort_values("ts").reset_index(drop=True)


def build_sym_frames(df5, vix_d, vix_5m):
    frames = {}
    for tf, rule in TFS.items():
        raw = df5 if tf == "5m" else s.resample(df5, rule)
        frames[tf] = p3.enrich(s.prep(raw, intraday=True))
        frames[tf] = vx.align_vix(
            frames[tf], vix_d, vix_5m if tf in ("5m", "15m") else None
        )
    return frames


def col(df, name):
    return df[name].fillna(False) if name in df.columns else pd.Series(False, index=df.index)


def long_recipes(df):
    b = s.base_mask(df, "bb_dn")
    first = b & ~b.shift(1).fillna(False)
    out = {
        "raw_bb_dn": first,
        "L1_prior_up": first & col(df, "prior_up"),
        "L2_hvol": first & col(df, "prior_up") & col(df, "high_vol"),
        "L3_rsi35": first & col(df, "prior_up") & col(df, "rsi35"),
    }
    if "vix5_rising" in df.columns and df["vix5_rising"].notna().any():
        out["L1v_vix5"] = out["L1_prior_up"] & col(df, "vix5_rising")
        out["L2v_hvol_vix5"] = out["L2_hvol"] & col(df, "vix5_rising")
        out["L3v_rsi35_vix5"] = out["L3_rsi35"] & col(df, "vix5_rising")
    if "vix_above_ma10" in df.columns:
        out["L1m_vix_ma10"] = out["L1_prior_up"] & col(df, "vix_above_ma10")
    return out


def short_recipes(df):
    b = s.base_mask(df, "bb_up")
    first = b & ~b.shift(1).fillna(False)
    out = {
        "raw_bb_up": first,
        "S_prior_dn_rsi65": first & col(df, "prior_down") & col(df, "rsi65"),
        "S_vwap_rsi65": first & col(df, "stretch_ok_short") & col(df, "rsi65"),
        "S_gap_up_stretch": first & col(df, "gap_up") & col(df, "stretch025"),
        "S_narrow_bb": first & col(df, "narrow_bb"),
    }
    if "vix5_crush" in df.columns and df["vix5_crush"].notna().any():
        out["Sv_vwap_rsi65_crush"] = out["S_vwap_rsi65"] & col(df, "vix5_crush")
    return out


def score(r):
    if r["n"] < 1 or np.isnan(r.get("avg", np.nan)):
        return -1e9
    return r["avg"] * np.sqrt(r["n"]) * (0.5 + r["wr"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock-25m", action="store_true")
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--symbols", nargs="+", default=SYMBOLS)
    args = ap.parse_args()

    holds_clock = {"5m": 5, "15m": 2, "30m": 1, "1h": 1}
    mode = "clock~25m" if args.clock_25m else "5bars"

    print("Loading VIX…")
    vix_d = vx.prep_vix_daily(vx.fetch_vix_daily())
    try:
        vix_5m = vx.prep_vix_5m(vx.fetch_vix_5m())
    except Exception as e:
        print(f"5m VIX skip: {e}")
        vix_5m = None

    all_rows = []
    for sym in args.symbols:
        print(f"\n{'=' * 88}\nSYMBOL {sym}\n{'=' * 88}")
        df5 = load_symbol_5m(sym)
        frames = build_sym_frames(df5, vix_d, vix_5m)
        for tf, df in frames.items():
            hold = holds_clock[tf] if args.clock_25m else 5
            print(f"\n-- {sym} {tf} hold={hold} --")
            for side, recipes in (("long", long_recipes(df)), ("short", short_recipes(df))):
                for name, mask in recipes.items():
                    _, r = s.backtest(df, mask, side, label=f"{sym}|{tf}|{name}", hold=hold)
                    row = {**r, "sym": sym, "tf": tf, "side": side, "recipe": name,
                           "hold": hold, "mode": mode}
                    all_rows.append(row)
                    if r["n"] >= max(5, args.min_n // 2):
                        print(f"  {side[0].upper()} {name:<22} n={r['n']:>3} "
                              f"({r['per_day']:.2f}/d) WR={r['wr']:.0%} "
                              f"avg={r['avg']:+.3%} med={r['med']:+.3%} "
                              f"MFE={r['mfe_med']:+.3%}")

    out = pd.DataFrame(all_rows)
    path = os.path.join(OUT, "signal_cross_asset_tf.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path} ({len(out)} rows)")

    # Survivors: n>=min_n, WR>=60%, avg>=+0.05%
    print("\n" + "=" * 88)
    print(f"SURVIVORS — {mode}, n>={args.min_n}, WR>=60%, avg>=+0.05%")
    print("=" * 88)
    surv = out[
        (out["n"] >= args.min_n)
        & (out["wr"] >= 0.60)
        & (out["avg"] >= 0.0005)
    ].copy()
    surv["score"] = surv.apply(score, axis=1)
    surv = surv.sort_values("score", ascending=False)
    for _, r in surv.head(40).iterrows():
        print(f"  {r.sym}|{r.tf}|{r.side}|{r.recipe:<22} "
              f"n={r.n:.0f} WR={r.wr:.0%} avg={r.avg:+.3%} MFE={r.mfe_med:+.3%}")
    surv.to_csv(os.path.join(OUT, "signal_cross_asset_survivors.csv"), index=False)

    # Cross-asset consistency for core recipes on 5m
    print("\n" + "=" * 88)
    print("5m CORE RECIPE — by symbol (n shown even if weak)")
    print("=" * 88)
    core = ["L1_prior_up", "L2_hvol", "L3_rsi35", "L1v_vix5", "L2v_hvol_vix5", "L3v_rsi35_vix5"]
    for recipe in core:
        print(f"\n  {recipe}:")
        sub = out[(out["tf"] == "5m") & (out["recipe"] == recipe) & (out["mode"] == mode)]
        for _, r in sub.sort_values("sym").iterrows():
            flag = "OK" if (r.n >= args.min_n and r.wr >= 0.6 and r.avg >= 0.0005) else ".."
            print(f"    [{flag}] {r.sym}: n={r.n:.0f} WR={r.wr:.0%} avg={r.avg:+.3%}")

    # Which symbols agree on 5m L-family?
    print("\n" + "=" * 88)
    print("AGREEMENT — symbols passing gate per 5m long recipe")
    print("=" * 88)
    for recipe in core:
        sub = surv[(surv["tf"] == "5m") & (surv["recipe"] == recipe) & (surv["side"] == "long")]
        syms = sorted(sub["sym"].unique())
        print(f"  {recipe:<18} {len(syms)}/4  {syms}")

    # TF heat for L1 across assets
    print("\n" + "=" * 88)
    print("TF HEAT — L1_prior_up avg by sym×tf (blank if n<min_n)")
    print("=" * 88)
    heat = out[(out["recipe"] == "L1_prior_up") & (out["side"] == "long")]
    for tf in ["5m", "15m", "30m", "1h"]:
        cells = []
        for sym in args.symbols:
            r = heat[(heat.tf == tf) & (heat.sym == sym)]
            if not len(r) or r.iloc[0]["n"] < args.min_n:
                cells.append(f"{sym}:—")
            else:
                rr = r.iloc[0]
                cells.append(f"{sym}:{rr['avg']:+.2%}@{rr['wr']:.0%}/n{rr['n']:.0f}")
        print(f"  {tf}:  " + "  ".join(cells))


if __name__ == "__main__":
    main()
