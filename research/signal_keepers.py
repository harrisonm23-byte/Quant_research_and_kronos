#!/usr/bin/env python3
"""Paper signal from phase-3 survivors — SPY BB-fade keepers.

LONG keepers (calls / long underlying proxy):
  L1  5m  bb_dn + prior_up                     (~0.8/d, WR~71%)
  L2  5m  bb_dn + prior_up + high_vol          (~0.65/d, WR~74%)
  L3  5m  bb_dn + prior_up + rsi35             (~0.4/d, WR~91%, selective)
  L4  15m bb_dn + stretch035                   (~0.23/d, WR~64%, big MFE)
  L5  15m bb_dn + rsi30                        (~0.27/d, WR~62%)

SHORT keepers (puts) — weaker / provisional:
  S1  15m bb_up + stretch_ok + rsi65
  S2  15m bb_up + gap_up + stretch025
  S3  5m  bb_up + prior_down + rsi65
  S4  15m bb_up + narrow_bb

Paper logging (multi-metric gate, not WR-only):
  python3 signal_keepers_paper.py gate|check|log|status

Usage:
  python3 signal_keepers.py           # backtest all keepers
  python3 signal_keepers.py --scan    # live armed state
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
import signal_combo_scan as s
import signal_combo_phase3 as p3

HOLD = 5


def masks(df, name):
    base_dn = s.base_mask(df, "bb_dn")
    base_up = s.base_mask(df, "bb_up")
    first_dn = base_dn & ~base_dn.shift(1).fillna(False)
    first_up = base_up & ~base_up.shift(1).fillna(False)
    f = df
    table = {
        "L1_5m_bbdn_prior_up": (first_dn & f["prior_up"].fillna(False), "long"),
        "L2_5m_bbdn_prior_up_hvol": (
            first_dn & f["prior_up"].fillna(False) & f["high_vol"].fillna(False), "long"),
        "L3_5m_bbdn_prior_up_rsi35": (
            first_dn & f["prior_up"].fillna(False) & f["rsi35"].fillna(False), "long"),
        "L4_15m_bbdn_stretch035": (
            first_dn & f["stretch035"].fillna(False), "long"),
        "L5_15m_bbdn_rsi30": (
            first_dn & f["rsi30"].fillna(False), "long"),
        "S1_15m_bbup_vwap_rsi65": (
            first_up & f["stretch_ok_short"].fillna(False) & f["rsi65"].fillna(False), "short"),
        "S2_15m_bbup_gap_up_stretch025": (
            first_up & f["gap_up"].fillna(False) & f["stretch025"].fillna(False), "short"),
        "S3_5m_bbup_prior_down_rsi65": (
            first_up & f["prior_down"].fillna(False) & f["rsi65"].fillna(False), "short"),
        "S4_15m_bbup_narrow_bb": (
            first_up & f["narrow_bb"].fillna(False), "short"),
    }
    return table[name]


def run_all(frames):
    specs = [
        ("5m", "L1_5m_bbdn_prior_up"),
        ("5m", "L2_5m_bbdn_prior_up_hvol"),
        ("5m", "L3_5m_bbdn_prior_up_rsi35"),
        ("15m", "L4_15m_bbdn_stretch035"),
        ("15m", "L5_15m_bbdn_rsi30"),
        ("15m", "S1_15m_bbup_vwap_rsi65"),
        ("15m", "S2_15m_bbup_gap_up_stretch025"),
        ("5m", "S3_5m_bbup_prior_down_rsi65"),
        ("15m", "S4_15m_bbup_narrow_bb"),
    ]
    rows, trades_all = [], []
    print("=" * 88)
    print("KEEPER BACKTEST — time exit (5 bars) and +0.15% target exit")
    print("=" * 88)
    for tf, name in specs:
        df = frames[tf]
        mask, side = masks(df, name)
        tr_t, r_t = s.backtest(df, mask, side, label=f"{name}|time", hold=HOLD)
        tr_g, r_g = p3.backtest_target(df, mask, side, hold=HOLD, target=0.0015)
        r_g["label"] = f"{name}|tgt15"
        print(f"\n{name} ({side})")
        print("  time:", s.fmt_row(r_t).strip())
        extra = f"  tgt_hit={r_g['tgt_rate']:.0%}" if r_g["n"] else ""
        print("  tgt :", s.fmt_row(r_g).strip() + extra)
        # walk-forward
        parts = {
            "L1_5m_bbdn_prior_up": ["prior_up"],
            "L2_5m_bbdn_prior_up_hvol": ["prior_up", "high_vol"],
            "L3_5m_bbdn_prior_up_rsi35": ["prior_up", "rsi35"],
            "L4_15m_bbdn_stretch035": ["stretch035"],
            "L5_15m_bbdn_rsi30": ["rsi30"],
            "S1_15m_bbup_vwap_rsi65": ["stretch_ok_short", "rsi65"],
            "S2_15m_bbup_gap_up_stretch025": ["gap_up", "stretch025"],
            "S3_5m_bbup_prior_down_rsi65": ["prior_down", "rsi65"],
            "S4_15m_bbup_narrow_bb": ["narrow_bb"],
        }[name]
        base = "bb_dn" if side == "long" else "bb_up"
        wf = p3.walkforward_check(df, base, parts, side)
        print(f"  WF  : H1 n={wf['H1']['n']} WR={wf['H1']['wr']:.0%} avg={wf['H1']['avg']:+.3%} | "
              f"H2 n={wf['H2']['n']} WR={wf['H2']['wr']:.0%} avg={wf['H2']['avg']:+.3%}")
        rows.append({**r_t, "keeper": name, "side": side, "tf": tf, "exit": "time"})
        rows.append({**r_g, "keeper": name, "side": side, "tf": tf, "exit": "tgt15"})
        if len(tr_t):
            tr_t = tr_t.copy()
            tr_t["keeper"] = name
            trades_all.append(tr_t)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT, "signal_keepers_summary.csv"), index=False)
    if trades_all:
        pd.concat(trades_all, ignore_index=True).to_csv(
            os.path.join(OUT, "signal_keepers_trades.csv"), index=False)
    print(f"\nWrote signal_keepers_summary.csv")
    return summary


def scan(frames):
    print("\n" + "=" * 88)
    print("SCAN — keeper armed state (last bars)")
    print("=" * 88)
    for tf in ("5m", "15m"):
        df = frames[tf]
        print(f"\n-- {tf} last 8 bars --")
        for _, r in df.tail(8).iterrows():
            flags = []
            if r.close <= r.bb_lo:
                flags.append("bb_dn")
            if r.close >= r.bb_up:
                flags.append("bb_up")
            if r.get("prior_up", False):
                flags.append("prior_up")
            if r.get("prior_down", False):
                flags.append("prior_down")
            if r.get("high_vol", False):
                flags.append("hvol")
            if r.get("rsi35", False):
                flags.append(f"RSI{r.rsi:.0f}")
            if r.get("rsi30", False):
                flags.append("rsi30")
            if r.get("rsi65", False):
                flags.append(f"RSI{r.rsi:.0f}")
            if r.get("stretch035", False):
                flags.append(f"vwap{r.vwap_dist:+.2%}")
            if r.get("stretch_ok_short", False):
                flags.append("vwap+")
            if r.get("tod_pm", False):
                flags.append("pm")
            print(f"  {r.ts.strftime('%m-%d %H:%M')} c={r.close:.2f} "
                  f"{'|'.join(flags) if flags else '-'}")

        # check which keepers fire on last bar
        last_i = df.index[-1]
        print("  armed now:")
        names_5m = [
            "L1_5m_bbdn_prior_up", "L2_5m_bbdn_prior_up_hvol", "L3_5m_bbdn_prior_up_rsi35",
            "S3_5m_bbup_prior_down_rsi65",
        ]
        names_15m = [
            "L4_15m_bbdn_stretch035", "L5_15m_bbdn_rsi30",
            "S1_15m_bbup_vwap_rsi65", "S2_15m_bbup_gap_up_stretch025", "S4_15m_bbup_narrow_bb",
        ]
        for name in (names_5m if tf == "5m" else names_15m):
            mask, side = masks(df, name)
            if bool(mask.iloc[-1]):
                print(f"    *** {name} ({side})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    df5 = s.load_5m()
    daily = s.load_daily()
    frames = s.build_frames(df5, daily)
    for tf in ["5m", "15m", "30m", "1h"]:
        frames[tf] = p3.enrich(frames[tf])
    run_all(frames)
    if args.scan:
        scan(frames)


if __name__ == "__main__":
    main()
