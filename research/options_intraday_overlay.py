#!/usr/bin/env python3
"""Modeled intraday options overlays for registered L1/L2/L3 strategies.

This is a structural Black-Scholes comparison, not a real-chain backtest.
Entry/exit timestamps and underlying prices come from the execution-correct
exit engine. IV is rolling 5m realized volatility times a stress multiplier;
the grid deliberately varies that multiplier. Premium costs use a 2% haircut.

Supported structures come from `intraday_strategy_registry.py`:
  * ATM long call, 2 DTE
  * ATM / +1% bull call spread, 2 DTE

Usage:
  python3 options_intraday_overlay.py --symbol QQQ --tag recent_60d
  python3 options_intraday_overlay.py --symbol QQQ \
    --source-file /path/to/QQQ_5m.csv --tag historical
  python3 options_intraday_overlay.py --symbol TQQQ --tag recent_60d
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

import signal_exit_mechanics as exits
import signal_htf_combo as htf
from intraday_strategy_registry import OVERLAYS, STRATEGIES

OUT = os.path.dirname(os.path.abspath(__file__))
R = 0.04
COST = 0.02
IV_MULTS = (1.0, 1.25, 1.5)


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(spot, strike, years, iv):
    if years <= 0:
        return max(spot - strike, 0.0)
    iv = max(iv, 0.01)
    d1 = (
        math.log(spot / strike) + (R + iv * iv / 2) * years
    ) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    return spot * normal_cdf(d1) - strike * math.exp(-R * years) * normal_cdf(d2)


def intraday_rv(panel):
    """Rolling ~20-session RV, excluding overnight jumps."""
    same_day = panel["day"].eq(panel["day"].shift(1))
    logret = np.log(panel["close"] / panel["close"].shift(1)).where(same_day)
    bars_per_year = 252 * 78
    return (
        logret.rolling(20 * 78, min_periods=5 * 78).std()
        * math.sqrt(bars_per_year)
    )


def iv_at_exit(iv0, underlying_ret):
    return iv0 * min(1.4, max(0.6, 1 - 3 * underlying_ret))


def overlay_strikes(overlay, entry_spot):
    k1 = entry_spot * (1 + overlay.moneyness)
    k2 = (
        entry_spot * (1 + overlay.moneyness + overlay.width)
        if overlay.width is not None else None
    )
    return k1, k2


def price_overlay(overlay, spot, years, iv, strikes=None):
    k1, k2 = strikes or overlay_strikes(overlay, spot)
    long_call = bs_call(spot, k1, years, iv)
    if overlay.structure == "long_call":
        return long_call
    if overlay.structure == "bull_call_spread":
        return max(long_call - bs_call(spot, k2, years, iv), 0.0)
    raise ValueError(overlay.structure)


def simulate(panel, trades, overlay, iv_mult):
    rv = intraday_rv(panel)
    ts_to_i = {pd.Timestamp(ts): i for i, ts in enumerate(panel["ts"])}
    pnls = []
    for trade in trades.itertuples():
        entry_ts = pd.Timestamp(trade.entry_ts)
        exit_ts = pd.Timestamp(trade.exit_ts)
        i = ts_to_i.get(entry_ts)
        j = ts_to_i.get(exit_ts)
        if i is None or j is None or not np.isfinite(rv.iloc[i]):
            continue
        s0, sx = float(trade.entry), float(trade.exit)
        iv0 = float(rv.iloc[i]) * iv_mult
        elapsed_days = max(j - i + 1, 0) / 78
        t0 = overlay.dte / 252
        tx = max(overlay.dte - elapsed_days, 0.02) / 252
        ive = iv_at_exit(iv0, sx / s0 - 1)
        strikes = overlay_strikes(overlay, s0)
        p0 = price_overlay(
            overlay, s0, t0, iv0, strikes=strikes
        ) * (1 + COST / 2)
        px = price_overlay(
            overlay, sx, tx, ive, strikes=strikes
        ) * (1 - COST / 2)
        if p0 < 0.05:
            continue
        pnls.append(px / p0 - 1)
    return np.asarray(pnls)


def summarize(symbol, setup, mechanic, overlay_id, iv_mult, pnl):
    return {
        "symbol": symbol,
        "setup": setup,
        "mechanic": mechanic,
        "overlay_id": overlay_id,
        "iv_mult": iv_mult,
        "n": len(pnl),
        "wr": (pnl > 0).mean() if len(pnl) else np.nan,
        "avg": pnl.mean() if len(pnl) else np.nan,
        "med": np.median(pnl) if len(pnl) else np.nan,
        "p10": np.quantile(pnl, 0.10) if len(pnl) else np.nan,
        "p05": np.quantile(pnl, 0.05) if len(pnl) else np.nan,
        "worst": pnl.min() if len(pnl) else np.nan,
        "premium_cap_usd": OVERLAYS[overlay_id].premium_cap_usd,
        "modeled_only": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", choices=["QQQ", "TQQQ"], required=True)
    ap.add_argument("--source-file")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    panel = htf.build_panel(args.symbol, source_file=args.source_file)
    masks, broad = exits.setup_masks(panel)
    rows = []

    for spec in STRATEGIES:
        if spec.symbol != args.symbol:
            continue
        mechanic = next(m for m in exits.MECHANICS if m[0] == spec.exit_mechanic)
        trades = exits.run_strategy(
            panel, masks[spec.setup_mask], broad, mechanic
        )
        for overlay_id in spec.overlay_ids:
            overlay = OVERLAYS[overlay_id]
            for iv_mult in IV_MULTS:
                pnl = simulate(panel, trades, overlay, iv_mult)
                rows.append(summarize(
                    args.symbol, spec.setup, spec.exit_mechanic,
                    overlay_id, iv_mult, pnl,
                ))

    result = pd.DataFrame(rows)
    path = os.path.join(
        OUT, f"options_intraday_overlay_{args.symbol}_{args.tag}.csv"
    )
    result.to_csv(path, index=False)
    print(f"Wrote {path} ({len(result)} rows)")
    print("MODELED ONLY: Black-Scholes + realized-vol proxy; no chain quotes/fills.\n")
    for _, r in result.sort_values(
        ["mechanic", "setup", "overlay_id", "iv_mult"]
    ).iterrows():
        print(
            f"{r['setup']:<2} {r['mechanic']:<9} {r['overlay_id']:<30} "
            f"IVx={r['iv_mult']:.2f} n={r['n']:>3} WR={r['wr']:.1%} "
            f"avg={r['avg']:+.1%} med={r['med']:+.1%} p05={r['p05']:+.1%}"
        )


if __name__ == "__main__":
    main()

