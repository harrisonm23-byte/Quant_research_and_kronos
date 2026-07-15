#!/usr/bin/env python3
"""Compare executable exit mechanics for 5m BB-fade keeper entries.

Properties deliberately different from the original event study:
  * one position at a time per setup;
  * entry at the next 5m open;
  * 1.5 bp slippage on both entry and exit;
  * every exit is forced before the session ends (no overnight leakage);
  * re-signal variants may extend the deadline, but never beyond 24 bars.

Mean-reversion exits use a bar-close confirmation and execute at the next
bar's open (never retroactively at the observed close). Target exits use the
bar high and fill at the target price; if a target and another exit happen in
the same bar, the target takes precedence. All variants have a same-session
cap.

Usage:
  python3 signal_exit_mechanics.py --symbol SPY --tag recent
  python3 signal_exit_mechanics.py --symbol SPY \
    --source-file /path/to/SPY_5_min.csv --tag historical_2019_2021
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import signal_htf_combo as htf

OUT = os.path.dirname(os.path.abspath(__file__))
SLIP = 0.00015  # 1.5 bp per side

SETUPS = [
    ("L1", "L1_prior_up", ()),
    ("L2", "L2_hvol", ()),
    ("L3", "L3_rsi35", ()),
    ("L1c", "L1_prior_up", ("15m_candle_dn",)),
    ("L2h", "L2_hvol", ("15m_candle_dn",)),
    ("L3h", "L3_rsi35", ("15m_candle_dn",)),
]

MECHANICS = [
    # name, maximum bars, close condition, target, re-signal extension
    ("fixed_5", 5, None, None, False),
    ("fixed_12", 12, None, None, False),
    ("sma9_close_max12", 12, "sma9", None, False),
    ("bbmid_close_max12", 12, "bb_mid", None, False),
    ("vwap_close_max12", 12, "vwap", None, False),
    ("target15_max12", 12, None, 0.0015, False),
    ("target25_max12", 12, None, 0.0025, False),
    ("target15_or_sma9_max12", 12, "sma9", 0.0015, False),
    ("resignal_extend5_cap24", 5, None, None, True),
    ("resignal_target15_or_sma9_cap24", 5, "sma9", 0.0015, True),
]


def setup_masks(df):
    recipes = htf.base_recipes(df)
    result = {}
    for label, recipe, flags in SETUPS:
        mask = recipes[recipe].copy()
        for flag in flags:
            if flag not in df:
                raise KeyError(f"{label}: missing {flag}")
            mask &= df[flag].fillna(False)
        result[label] = mask.fillna(False)
    return result, recipes["L1_prior_up"].fillna(False)


def session_last_indices(df):
    """Map each row to the final row index of its session."""
    return (
        df.groupby("day", sort=False)["ts"]
        .transform(lambda x: x.index[-1])
        .astype(int)
        .to_numpy()
    )


def run_strategy(df, signal, broad_resignal, mechanic):
    name, initial_bars, close_col, target, extend = mechanic
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    threshold = (
        df[close_col].to_numpy(float) if close_col is not None else None
    )
    sig = signal.to_numpy(bool)
    re_sig = broad_resignal.to_numpy(bool)
    last_of_day = session_last_indices(df)
    trades = []
    next_available = 0

    for i in np.flatnonzero(sig):
        if i < next_available:
            continue
        entry_i = i + 1
        if entry_i >= len(df) or entry_i > last_of_day[i]:
            continue  # no next-open entry in this session

        entry = o[entry_i] * (1 + SLIP)
        hard_cap = min(entry_i + 23, last_of_day[i])
        deadline = min(entry_i + initial_bars - 1, last_of_day[i])
        exit_i = deadline
        exit_raw = close[deadline]
        path_last_i = deadline
        reason = "time" if deadline < last_of_day[i] else "eod"
        j = entry_i

        while j <= deadline:
            target_hit = target is not None and h[j] >= entry * (1 + target)
            close_hit = (
                threshold is not None
                and np.isfinite(threshold[j])
                and close[j] >= threshold[j]
            )
            if target_hit:
                exit_i = j
                exit_raw = entry * (1 + target)
                path_last_i = j
                reason = f"target_{int(target * 10000)}bp"
                break
            # The crossing is only known after bar j closes. Execute at the
            # next open, and only when j is before the time/session deadline.
            if close_hit and j < deadline and j < last_of_day[i]:
                exit_i = j + 1
                exit_raw = o[j + 1]
                path_last_i = j  # do not count post-entry-bar high/low
                reason = f"{close_col}_close"
                break

            # A subsequent broad L1 event resets the five-bar clock. The
            # absolute 24-bar/session cap prevents indefinite goalpost moving.
            if extend and re_sig[j]:
                deadline = min(max(deadline, j + 5), hard_cap)
                reason = "resignal_time"
            j += 1

        # If extension changed the deadline and no condition fired, use it.
        if j > deadline:
            exit_i = deadline
            exit_raw = close[deadline]
            path_last_i = deadline
            if deadline == last_of_day[i]:
                reason = "eod"

        exit_px = exit_raw * (1 - SLIP)
        path_h = h[entry_i:path_last_i + 1]
        path_l = low[entry_i:path_last_i + 1]
        # A next-open exit can gap beyond the prior bar's range.
        observed_high = max(path_h.max(), exit_raw)
        observed_low = min(path_l.min(), exit_raw)
        trades.append({
            "signal_ts": df["ts"].iloc[i],
            "entry_ts": df["ts"].iloc[entry_i],
            "exit_ts": df["ts"].iloc[exit_i],
            "day": df["day"].iloc[i],
            "entry": entry,
            "exit": exit_px,
            "ret": exit_px / entry - 1,
            "mfe": observed_high / entry - 1,
            "mae": observed_low / entry - 1,
            "bars": exit_i - entry_i + 1,
            "exit_reason": reason,
            "mechanic": name,
        })
        # A signal on the exit bar can enter at the following open.
        next_available = exit_i

    return pd.DataFrame(trades)


def summarize(trades, setup, mechanic, window):
    if len(trades):
        reasons = trades["exit_reason"].value_counts(normalize=True)
        reason_text = ";".join(f"{k}:{v:.1%}" for k, v in reasons.items())
    else:
        reason_text = ""
    return {
        "setup": setup,
        "mechanic": mechanic,
        "window": window,
        "n": len(trades),
        "wr": (trades["ret"] > 0).mean() if len(trades) else np.nan,
        "avg": trades["ret"].mean() if len(trades) else np.nan,
        "med": trades["ret"].median() if len(trades) else np.nan,
        "sum": trades["ret"].sum() if len(trades) else 0.0,
        "mfe_med": trades["mfe"].median() if len(trades) else np.nan,
        "mae_med": trades["mae"].median() if len(trades) else np.nan,
        "avg_bars": trades["bars"].mean() if len(trades) else np.nan,
        "p_hit15": (trades["mfe"] >= 0.0015).mean() if len(trades) else np.nan,
        "p_hit25": (trades["mfe"] >= 0.0025).mean() if len(trades) else np.nan,
        "exit_reasons": reason_text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--source-file")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    sym = args.symbol.upper()
    df = htf.build_panel(sym, source_file=args.source_file)
    masks, broad_resignal = setup_masks(df)
    midpoint = df["ts"].quantile(0.5)
    rows = []
    all_trades = []

    for setup, signal in masks.items():
        for mechanic in MECHANICS:
            tr = run_strategy(df, signal, broad_resignal, mechanic)
            if len(tr):
                tr.insert(0, "setup", setup)
                all_trades.append(tr)
            rows.append(summarize(tr, setup, mechanic[0], "full"))
            if len(tr):
                rows.append(summarize(
                    tr[tr["signal_ts"] <= midpoint], setup, mechanic[0], "H1"
                ))
                rows.append(summarize(
                    tr[tr["signal_ts"] > midpoint], setup, mechanic[0], "H2"
                ))

    summary = pd.DataFrame(rows)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    prefix = os.path.join(OUT, f"signal_exit_mechanics_{sym}_{args.tag}")
    summary.to_csv(prefix + "_summary.csv", index=False)
    trades.to_csv(prefix + "_trades.csv", index=False)
    print(f"Wrote {prefix}_summary.csv ({len(summary)} rows)")
    print(f"Wrote {prefix}_trades.csv ({len(trades)} rows)")

    full = summary[summary["window"] == "full"].copy()
    halves = summary[summary["window"].isin(["H1", "H2"])].pivot(
        index=["setup", "mechanic"], columns="window", values=["n", "wr", "avg"]
    )
    full["robust"] = full.apply(
        lambda r: (
            (r["setup"], r["mechanic"]) in halves.index
            and halves.loc[(r["setup"], r["mechanic"]), ("n", "H1")] >= 20
            and halves.loc[(r["setup"], r["mechanic"]), ("n", "H2")] >= 20
            and halves.loc[(r["setup"], r["mechanic"]), ("avg", "H1")] > 0
            and halves.loc[(r["setup"], r["mechanic"]), ("avg", "H2")] > 0
        ),
        axis=1,
    )
    ranked = full.sort_values(["robust", "avg"], ascending=False)
    print("\nFULL-PERIOD EXIT MATRIX (two-sided slippage, no overnight)")
    print("=" * 110)
    for _, r in ranked.iterrows():
        print(
            f"{r['setup']:<4} {r['mechanic']:<35} n={r['n']:>3} "
            f"WR={r['wr']:.1%} avg={r['avg']:+.3%} med={r['med']:+.3%} "
            f"bars={r['avg_bars']:.1f} robust_halves={'Y' if r['robust'] else 'N'}"
        )


if __name__ == "__main__":
    main()
