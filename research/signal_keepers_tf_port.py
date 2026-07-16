#!/usr/bin/env python3
"""Port L1/L2/L3 recipes across timeframes (5m / 15m / 30m / 1h).

Primary keepers were fit on 5m. This checks whether prior_up + bb_dn
(+ hvol / rsi35 / vix5) still works when the bar size changes.

Holds:
  default: 5 bars on each TF
  --clock-25m: ~25m clock hold (5@5m, 2@15m, 1@30m/1h)

Usage:
  python3 signal_keepers_tf_port.py
  python3 signal_keepers_tf_port.py --clock-25m
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
import signal_combo_scan as s
import signal_combo_phase3 as p3
import signal_vix_study as vx


def recipes(df):
    b = s.base_mask(df, "bb_dn")
    first = b & ~b.shift(1).fillna(False)

    def c(name):
        return df[name].fillna(False) if name in df.columns else pd.Series(False, index=df.index)

    return {
        "L1 prior_up": first & c("prior_up"),
        "L2 +hvol": first & c("prior_up") & c("high_vol"),
        "L3 +rsi35": first & c("prior_up") & c("rsi35"),
        "L1v +vix5up": first & c("prior_up") & c("vix5_rising"),
        "L2v +hvol+vix5": first & c("prior_up") & c("high_vol") & c("vix5_rising"),
        "L3v +rsi35+vix5": first & c("prior_up") & c("rsi35") & c("vix5_rising"),
        "raw bb_dn": first,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock-25m", action="store_true",
                    help="Use ~25m clock hold instead of 5 bars/TF")
    args = ap.parse_args()

    df5 = s.load_5m()
    daily = s.load_daily()
    frames = s.build_frames(df5, daily)
    vix_d = vx.prep_vix_daily(vx.fetch_vix_daily())
    try:
        vix_5m = vx.prep_vix_5m(vx.fetch_vix_5m())
    except Exception:
        vix_5m = None
    for tf in ["5m", "15m", "30m", "1h"]:
        frames[tf] = p3.enrich(frames[tf])
        frames[tf] = vx.align_vix(frames[tf], vix_d, vix_5m if tf in ("5m", "15m") else None)

    holds = {"5m": 5, "15m": 2, "30m": 1, "1h": 1} if args.clock_25m else {t: 5 for t in ["5m", "15m", "30m", "1h"]}
    mode = "clock~25m" if args.clock_25m else "5 bars/TF"
    print(f"Mode={mode}")
    print(f"{'TF':<4} {'recipe':<22} {'hold':>4} {'n':>4} {'/d':>5} {'WR':>5} {'avg':>8} {'med':>8} {'MFE':>8}")
    rows = []
    for tf in ["5m", "15m", "30m", "1h"]:
        df = frames[tf]
        hold = holds[tf]
        for name, mask in recipes(df).items():
            if "vix5" in name and ("vix5_rising" not in df.columns or df["vix5_rising"].isna().all()):
                continue
            _, r = s.backtest(df, mask, "long", label=f"{tf}|{name}", hold=hold)
            if r["n"] == 0:
                continue
            print(f"{tf:<4} {name:<22} {hold:>4} {r['n']:>4} {r['per_day']:>5.2f} "
                  f"{r['wr']:>4.0%} {r['avg']:>+7.3%} {r['med']:>+7.3%} {r['mfe_med']:>+7.3%}")
            rows.append({**r, "tf": tf, "recipe": name, "hold": hold, "mode": mode})
    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "signal_keepers_tf_port.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")
    board = out[out["n"] >= 10].sort_values("avg", ascending=False)
    print(f"\nTOP (n>=10) by avg — {mode}:")
    for _, r in board.head(12).iterrows():
        print(f"  {r.tf}|{r.recipe:<22} n={r.n:.0f} WR={r.wr:.0%} avg={r.avg:+.3%}")


if __name__ == "__main__":
    main()
