#!/usr/bin/env python3
"""Conditional day-structure probabilities.

Questions:
  1. If yesterday closed down AND today opens down (gap down), does the day
     tend to recover?  Recovery measured three ways:
       * close > open        (intraday recovery)
       * close > prior close (full recovery above yesterday's close)
       * high >= prior close (gap fills at some point)
  2. Either way, WHEN does the session high (and low) print?

Daily stats use {SYM}_daily.csv (10y).  Timing stats use 5m files.
No lookahead: conditions are known at the open; outcomes measured after.

Usage:
  python3 conditional_day_structure.py                       # daily SPY+QQQ
  python3 conditional_day_structure.py --intraday SPY_5m_yf.csv --symbol SPY --tag recent
  python3 conditional_day_structure.py --intraday /tmp/spy-intraday/SPY_5_min.csv \
      --symbol SPY --tag hist_2019_2021
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)

CONDITIONS = [
    ("dn_close_dn_open", "prior day down, opens down"),
    ("dn_close_up_open", "prior day down, opens up"),
    ("up_close_dn_open", "prior day up, opens down"),
    ("up_close_up_open", "prior day up, opens up"),
]


def classify(prior_down, gap_down):
    if prior_down and gap_down:
        return "dn_close_dn_open"
    if prior_down and not gap_down:
        return "dn_close_up_open"
    if not prior_down and gap_down:
        return "up_close_dn_open"
    return "up_close_up_open"


# ---------------------------------------------------------------- daily part
def daily_stats(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"))
    col = next(c for c in df.columns if c.lower() in ("date", "ts", "timestamps"))
    df["date"] = pd.to_datetime(df[col])
    df = df.sort_values("date").reset_index(drop=True)
    df["prior_close"] = df["close"].shift(1)
    df["prior_down"] = df["close"].shift(1) < df["close"].shift(2)
    df["gap_down"] = df["open"] < df["prior_close"]
    df = df.dropna(subset=["prior_close", "prior_down"]).reset_index(drop=True)
    df["cond"] = [classify(p, g) for p, g in zip(df["prior_down"], df["gap_down"])]

    rows = []
    for cond, label in CONDITIONS:
        sub = df[df["cond"] == cond]
        if not len(sub):
            continue
        oc = sub["close"] / sub["open"] - 1
        cc = sub["close"] / sub["prior_close"] - 1
        rows.append({
            "symbol": sym, "cond": cond, "label": label, "n": len(sub),
            "p_close_above_open": (sub["close"] > sub["open"]).mean(),
            "p_close_above_prior": (sub["close"] > sub["prior_close"]).mean(),
            "p_gap_fill": (sub["high"] >= sub["prior_close"]).mean(),
            "avg_open_to_close": oc.mean(),
            "med_open_to_close": oc.median(),
            "avg_close_to_close": cc.mean(),
            "avg_gap": (sub["open"] / sub["prior_close"] - 1).mean(),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- intraday part
def load_5m_any(path_or_name, sym):
    import signal_htf_combo as htf
    path = path_or_name
    if not os.path.isabs(path):
        cand = os.path.join(OUT, path)
        path = cand if os.path.exists(cand) else path_or_name
    return htf.load_5m(sym, source_file=path)


BUCKETS = [
    ("09:30-10:00", 0, 30), ("10:00-10:30", 30, 60), ("10:30-11:00", 60, 90),
    ("11:00-12:00", 90, 150), ("12:00-13:00", 150, 210),
    ("13:00-14:00", 210, 270), ("14:00-15:00", 270, 330),
    ("15:00-16:00", 330, 391),
]


def bucket_of(minutes):
    for name, lo, hi in BUCKETS:
        if lo <= minutes < hi:
            return name
    return BUCKETS[-1][0]


def intraday_stats(df5, sym, tag):
    df5 = df5.copy()
    df5["day"] = df5["ts"].dt.date
    df5["mins"] = (
        (df5["ts"].dt.hour - 9) * 60 + df5["ts"].dt.minute - 30
    )
    days = []
    for day, grp in df5.groupby("day", sort=True):
        hi_i = grp["high"].idxmax()
        lo_i = grp["low"].idxmin()
        days.append({
            "day": day,
            "open": grp["open"].iloc[0],
            "close": grp["close"].iloc[-1],
            "high": grp["high"].max(),
            "low": grp["low"].min(),
            "hi_min": grp.loc[hi_i, "mins"],
            "lo_min": grp.loc[lo_i, "mins"],
        })
    d = pd.DataFrame(days).sort_values("day").reset_index(drop=True)
    d["prior_close"] = d["close"].shift(1)
    d["prior_down"] = d["close"].shift(1) < d["close"].shift(2)
    d["gap_down"] = d["open"] < d["prior_close"]
    d = d.dropna(subset=["prior_close", "prior_down"]).reset_index(drop=True)
    d["cond"] = [classify(p, g) for p, g in zip(d["prior_down"], d["gap_down"])]
    d["recovered"] = d["close"] > d["open"]
    d["hi_bucket"] = d["hi_min"].map(bucket_of)
    d["lo_bucket"] = d["lo_min"].map(bucket_of)

    out_rows = []
    print(f"\n{'='*96}\nINTRADAY TIMING — {sym} [{tag}]  ({len(d)} conditioned sessions)")
    print("="*96)
    for cond, label in CONDITIONS:
        sub = d[d["cond"] == cond]
        if len(sub) < 5:
            continue
        for split_name, split in [
            ("all", sub),
            ("recovered", sub[sub["recovered"]]),
            ("failed", sub[~sub["recovered"]]),
        ]:
            if len(split) < 5:
                continue
            hi_dist = split["hi_bucket"].value_counts(normalize=True)
            lo_dist = split["lo_bucket"].value_counts(normalize=True)
            row = {
                "symbol": sym, "tag": tag, "cond": cond, "split": split_name,
                "n": len(split),
                "p_recovered": split["recovered"].mean() if split_name == "all" else np.nan,
                "med_high_min": split["hi_min"].median(),
                "med_low_min": split["lo_min"].median(),
                "p_high_first30": (split["hi_min"] < 30).mean(),
                "p_high_first60": (split["hi_min"] < 60).mean(),
                "p_high_last60": (split["hi_min"] >= 330).mean(),
                "p_low_first30": (split["lo_min"] < 30).mean(),
                "p_low_first60": (split["lo_min"] < 60).mean(),
                "p_low_last60": (split["lo_min"] >= 330).mean(),
            }
            for name, _, _ in BUCKETS:
                row[f"hi_{name}"] = hi_dist.get(name, 0.0)
                row[f"lo_{name}"] = lo_dist.get(name, 0.0)
            out_rows.append(row)
        # console table for this condition
        a = [r for r in out_rows if r["cond"] == cond and r["symbol"] == sym]
        print(f"\n-- {label} --")
        for r in a:
            rec = f" P(recover)={r['p_recovered']:.0%}" if r["split"] == "all" else ""
            hh, mm = divmod(int(r["med_high_min"]), 60)
            lh, lm = divmod(int(r["med_low_min"]), 60)
            print(
                f"  {r['split']:<10} n={r['n']:>3}{rec}"
                f"  med HIGH {9+ (30+r['med_high_min'])//60:02.0f}:{int((30+r['med_high_min'])%60):02d}"
                f"  med LOW {9+ (30+r['med_low_min'])//60:02.0f}:{int((30+r['med_low_min'])%60):02d}"
                f"  P(high 1st30m)={r['p_high_first30']:.0%}"
                f"  P(high last hr)={r['p_high_last60']:.0%}"
                f"  P(low 1st30m)={r['p_low_first30']:.0%}"
            )
    return pd.DataFrame(out_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intraday", help="5m csv (name in research/ or abs path)")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--tag", default="daily")
    args = ap.parse_args()

    if not args.intraday:
        allrows = []
        for sym in ("SPY", "QQQ"):
            t = daily_stats(sym)
            allrows.append(t)
            print(f"\n{'='*96}\nDAILY CONDITIONALS — {sym} (2016-2026)\n{'='*96}")
            for _, r in t.iterrows():
                print(
                    f"  {r['label']:<28} n={r['n']:>4}"
                    f"  P(close>open)={r['p_close_above_open']:.1%}"
                    f"  P(close>prior)={r['p_close_above_prior']:.1%}"
                    f"  P(gap fill)={r['p_gap_fill']:.1%}"
                    f"  avg O->C={r['avg_open_to_close']:+.3%}"
                    f"  med={r['med_open_to_close']:+.3%}"
                )
        out = pd.concat(allrows, ignore_index=True)
        path = os.path.join(OUT, "conditional_day_daily.csv")
        out.to_csv(path, index=False)
        print(f"\nWrote {path}")
        return

    df5 = load_5m_any(args.intraday, args.symbol)
    res = intraday_stats(df5, args.symbol, args.tag)
    path = os.path.join(
        OUT, f"conditional_day_timing_{args.symbol}_{args.tag}.csv"
    )
    res.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
