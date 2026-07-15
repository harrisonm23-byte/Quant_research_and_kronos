#!/usr/bin/env python3
"""Directly validate promoted HTF motifs on an explicit historical 5m file.

Unlike the discovery scan, this script does not seed or search combinations:
it tests a fixed, predeclared shortlist and reports full-period, half-sample,
and calendar-year statistics.  That makes it suitable for honest validation
on data that was not used to discover the current keeper set.

Usage:
  python3 signal_htf_regime_validation.py \
    --symbol SPY --source-file /path/to/SPY_5_min.csv \
    --tag historical_2019_2021

Historical dataset used for the checked-in report:
  https://www.kaggle.com/datasets/abidou/spy-intraday-ohlc
  File: SPY_5_min.csv, 2019-12-30 through 2021-10-28.
The publisher does not declare a license in Kaggle's API metadata, so the raw
bars are intentionally not redistributed in this repository.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

import signal_combo_scan as s
import signal_htf_combo as htf

OUT = os.path.dirname(os.path.abspath(__file__))
HOLD = 5

# Baselines plus the simple/pair motifs promoted from the recent cross-asset
# scan.  Test these directly even when a historical discovery seed would omit
# one of their constituent flags.
CANDIDATES = [
    ("raw_bb_dn", ()),
    ("L1_prior_up", ()),
    ("L2_hvol", ()),
    ("L3_rsi35", ()),
    ("L1_prior_up", ("15m_candle_dn",)),
    ("L1_prior_up", ("1w_above_sma9",)),
    ("L1_prior_up", ("15m_candle_dn", "1w_above_sma9")),
    ("L2_hvol", ("15m_candle_dn",)),
    ("L2_hvol", ("1h_bb_mid", "1w_above_sma9")),
    ("L3_rsi35", ("15m_candle_dn",)),
    ("L3_rsi35", ("1h_above_sma9",)),
    ("L3_rsi35", ("1w_above_sma9",)),
    ("L3_rsi35", ("15m_below_sma9", "1w_above_sma9")),
    ("L3_rsi35", ("15m_candle_dn", "1h_bb_mid")),
]


def windows(df):
    """Named, non-overlapping diagnostic windows."""
    midpoint = df["ts"].quantile(0.5)
    out = [
        ("full", pd.Series(True, index=df.index)),
        ("H1", df["ts"] <= midpoint),
        ("H2", df["ts"] > midpoint),
    ]
    for year in sorted(df["ts"].dt.year.unique()):
        mask = df["ts"].dt.year == year
        if mask.sum() >= 100:
            out.append((str(year), mask))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--source-file", required=True)
    ap.add_argument("--tag", default="historical")
    args = ap.parse_args()

    sym = args.symbol.upper()
    df = htf.build_panel(sym, source_file=args.source_file)
    recipes = htf.base_recipes(df)
    spans = windows(df)
    rows = []

    for recipe, flags in CANDIDATES:
        mask = recipes[recipe].copy()
        missing = [f for f in flags if f not in df.columns]
        if missing:
            raise KeyError(f"{recipe}: missing HTF flags {missing}")
        for flag in flags:
            mask &= df[flag].fillna(False)
        combo = "+".join(flags) if flags else "alone"

        for window, selected in spans:
            _, r = s.backtest(
                df, mask & selected, "long",
                label=f"{sym}|{recipe}|{combo}|{window}", hold=HOLD,
            )
            rows.append({
                "symbol": sym,
                "recipe": recipe,
                "combo": combo,
                "window": window,
                "start": df.loc[selected, "ts"].min(),
                "end": df.loc[selected, "ts"].max(),
                **{k: r[k] for k in (
                    "n", "wr", "avg", "med", "sum", "mfe_med", "hit15", "hit25"
                )},
            })

    result = pd.DataFrame(rows)
    path = os.path.join(
        OUT, f"signal_htf_regime_validation_{sym}_{args.tag}.csv"
    )
    result.to_csv(path, index=False)
    print(f"Wrote {path} ({len(result)} rows)")

    full = result[result["window"] == "full"]
    print("\nFULL-PERIOD DIRECT VALIDATION")
    print("=" * 100)
    for _, r in full.iterrows():
        print(
            f"{r['recipe']:<13} {r['combo']:<43} "
            f"n={r['n']:>4} WR={r['wr']:.1%} avg={r['avg']:+.3%} "
            f"med={r['med']:+.3%}"
        )

    print("\nHALVES / YEARS")
    print("=" * 100)
    detail = result[result["window"] != "full"]
    for (recipe, combo), group in detail.groupby(["recipe", "combo"], sort=False):
        stats = " | ".join(
            f"{r.window}: n={r.n} WR={r.wr:.0%} avg={r.avg:+.3%}"
            for r in group.itertuples()
        )
        print(f"{recipe}+{combo}: {stats}")


if __name__ == "__main__":
    main()
