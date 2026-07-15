#!/usr/bin/env python3
"""Phase 3 — deep filter search on shortlisted bases.

Bases (from phases 1–2):
  LONG:  5m|bb_dn, 15m|bb_dn
  SHORT: 5m|bb_up, 15m|bb_up  (+ seed with vwap+rsi65)

New filters under test:
  tod_am / tod_mid / tod_pm     session buckets (ET)
  gap_down / gap_up            vs prior RTH close
  prior_down / prior_up        prior day close < / > prior open
  narrow_bb / wide_bb          BB width percentile within day/rolling
  high_vol / low_vol           volume vs 20-bar median
  macd_turn                    MACD hist peaked/troughed in last 3 bars
  stretch025 / stretch035      |vwap_dist| thresholds
  rsi70 / rsi30                stronger RSI confirm

Also tests delayed SMA9 trigger after base, and target-exit proxy
(exit at +0.15% MFE if hit within hold, else time).

Usage:
  python3 signal_combo_phase3.py
  python3 signal_combo_phase3.py --min-n 12
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)

import signal_combo_scan as s  # noqa: E402

HOLD = 5
MIN_N_DEFAULT = 12


def enrich(df, daily=None):
    """Add phase-3 features onto a prepared TF frame."""
    df = df.copy()
    hhmm = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["tod_am"] = (hhmm >= 9 * 60 + 30) & (hhmm < 11 * 60)      # 09:30–11:00
    df["tod_mid"] = (hhmm >= 11 * 60) & (hhmm < 14 * 60)         # 11:00–14:00
    df["tod_pm"] = (hhmm >= 14 * 60) & (hhmm <= 15 * 60 + 55)    # 14:00–close

    df["bb_width"] = (df["bb_up"] - df["bb_lo"]) / df["close"]
    # rolling width rank over ~2 sessions of bars (adaptive)
    win = min(78, max(20, len(df) // 10))
    df["bb_width_pct"] = df["bb_width"].rolling(win).rank(pct=True)
    df["narrow_bb"] = df["bb_width_pct"] <= 0.35
    df["wide_bb"] = df["bb_width_pct"] >= 0.65

    vol_med = df["volume"].rolling(20).median().replace(0, np.nan)
    df["vol_ratio"] = df["volume"] / vol_med
    df["high_vol"] = df["vol_ratio"] >= 1.25
    df["low_vol"] = df["vol_ratio"] <= 0.80

    # MACD turn: hist made a local extreme then reversed
    mh = df["macdh"]
    df["macd_turn_short"] = (mh.shift(1) > mh.shift(2)) & (mh < mh.shift(1))  # peaked
    df["macd_turn_long"] = (mh.shift(1) < mh.shift(2)) & (mh > mh.shift(1))   # troughed

    df["stretch025"] = df["vwap_dist"].abs() >= 0.0025
    df["stretch035"] = df["vwap_dist"].abs() >= 0.0035
    df["stretch_ok_short"] = df["vwap_dist"] >= 0.0020
    df["stretch_ok_long"] = df["vwap_dist"] <= -0.0020
    df["rsi70"] = df["rsi"] >= 70
    df["rsi30"] = df["rsi"] <= 30
    df["rsi65"] = df["rsi"] >= 65
    df["rsi35"] = df["rsi"] <= 35

    # day-level context from 5m/daily closes
    day_ohlc = df.groupby("day").agg(
        d_open=("open", "first"),
        d_close=("close", "last"),
    )
    day_ohlc["prior_close"] = day_ohlc["d_close"].shift(1)
    day_ohlc["prior_open"] = day_ohlc["d_open"].shift(1)
    day_ohlc["gap"] = day_ohlc["d_open"] / day_ohlc["prior_close"] - 1.0
    day_ohlc["prior_ret"] = day_ohlc["d_close"] / day_ohlc["d_open"] - 1.0
    day_ohlc["gap_down"] = day_ohlc["gap"] <= -0.0015
    day_ohlc["gap_up"] = day_ohlc["gap"] >= 0.0015
    day_ohlc["prior_down"] = day_ohlc["prior_ret"] < 0
    day_ohlc["prior_up"] = day_ohlc["prior_ret"] > 0
    for c in ["gap_down", "gap_up", "prior_down", "prior_up", "gap", "prior_ret"]:
        df[c] = df["day"].map(day_ohlc[c])

    return df


def filter_lib(side):
    """Named boolean columns available as filters for this side."""
    common = ["tod_am", "tod_mid", "tod_pm", "narrow_bb", "wide_bb",
              "high_vol", "low_vol", "gap_down", "gap_up",
              "prior_down", "prior_up"]
    if side == "short":
        return common + ["stretch_ok_short", "stretch025", "stretch035",
                         "rsi65", "rsi70", "macd_turn_short"]
    return common + ["stretch_ok_long", "stretch025", "stretch035",
                     "rsi35", "rsi30", "macd_turn_long"]


def backtest_target(df, entry_mask, side, hold=HOLD, target=0.0015, dedup=5):
    """Exit at target if touched within hold, else close of hold bar."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    mask = entry_mask.fillna(False).values
    n = len(df)
    trades, last = [], -10**9
    for i in range(30, n - hold - 1):
        if not mask[i] or i - last < dedup:
            continue
        entry = o[i + 1] * (1 + s.SLIP if side == "long" else 1 - s.SLIP)
        exit_px, reason, held = None, "time", hold
        for k in range(1, hold + 1):
            j = i + k
            if j >= n:
                break
            held = k
            if side == "long":
                if h[j] >= entry * (1 + target):
                    exit_px, reason = entry * (1 + target), "target"
                    break
            else:
                if l[j] <= entry * (1 - target):
                    exit_px, reason = entry * (1 - target), "target"
                    break
        if exit_px is None:
            exit_px = c[min(i + hold, n - 1)]
        if side == "short":
            ret = entry / exit_px - 1.0
            mfe = entry / l[i + 1:i + 1 + held].min() - 1.0
        else:
            ret = exit_px / entry - 1.0
            mfe = h[i + 1:i + 1 + held].max() / entry - 1.0
        trades.append(dict(ret=ret, mfe=mfe, reason=reason, held=held,
                           ts=df["ts"].iloc[i], day=df["day"].iloc[i]))
        last = i
    tr = pd.DataFrame(trades)
    n_days = max(int(df["day"].nunique()), 1)
    if not len(tr):
        return tr, dict(n=0, per_day=0, wr=np.nan, avg=np.nan, med=np.nan,
                        sum=0.0, mfe_med=np.nan, hit15=np.nan, hit25=np.nan,
                        tgt_rate=np.nan)
    return tr, dict(
        n=len(tr), per_day=len(tr) / n_days,
        wr=float((tr["ret"] > 0).mean()),
        avg=float(tr["ret"].mean()),
        med=float(tr["ret"].median()),
        sum=float(tr["ret"].sum()),
        mfe_med=float(tr["mfe"].median()),
        hit15=float((tr["mfe"] >= 0.0015).mean()),
        hit25=float((tr["mfe"] >= 0.0025).mean()),
        tgt_rate=float((tr["reason"] == "target").mean()),
    )


def score(row):
    if row["n"] < 1 or np.isnan(row.get("avg", np.nan)):
        return -1e9
    # favor positive avg with sample size; penalize low WR
    return row["avg"] * np.sqrt(row["n"]) * (0.5 + row["wr"])


def search_base(df, base_name, tf, min_n=MIN_N_DEFAULT, max_filters=3):
    side = s.side_of(base_name)
    base = s.base_mask(df, base_name)
    first = base & ~base.shift(1).fillna(False)
    filters = filter_lib(side)

    rows = []

    def record(label, mask, mode="time"):
        if mode == "time":
            _, r = s.backtest(df, mask, side, label=label, hold=HOLD)
        else:
            _, r = backtest_target(df, mask, side, hold=HOLD, target=0.0015)
            r["label"] = label
        r.update(dict(tf=tf, base=base_name, side=side, mode=mode,
                      filters=label.split("|", 2)[-1] if "|" in label else label))
        rows.append(r)

    # baseline
    record(f"{tf}|{base_name}|alone", first, "time")
    record(f"{tf}|{base_name}|alone@tgt", first, "target")

    # delayed SMA9 after extreme
    delayed = s.delayed_sma9_mask(df, first, side, look=8)
    record(f"{tf}|{base_name}|>>sma9", delayed, "time")
    record(f"{tf}|{base_name}|>>sma9@tgt", delayed, "target")

    # singles
    for f in filters:
        m = first & df[f].fillna(False)
        record(f"{tf}|{base_name}|{f}", m, "time")

    # pairs
    for a, b in itertools.combinations(filters, 2):
        # skip contradictory TOD / gap / prior pairs
        if {a, b} <= {"tod_am", "tod_mid", "tod_pm"}:
            continue
        if {a, b} == {"gap_down", "gap_up"}:
            continue
        if {a, b} == {"prior_down", "prior_up"}:
            continue
        if {a, b} == {"narrow_bb", "wide_bb"}:
            continue
        if {a, b} == {"high_vol", "low_vol"}:
            continue
        m = first & df[a].fillna(False) & df[b].fillna(False)
        record(f"{tf}|{base_name}|{a}+{b}", m, "time")

    # triples on promising singles only (top by avg among n>=min_n/2)
    singles = [r for r in rows if r["mode"] == "time" and r["n"] >= max(8, min_n // 2)
               and "+" not in str(r.get("filters", "")) and r.get("filters") not in
               ("alone", ">>sma9", "alone@tgt", ">>sma9@tgt")]
    singles = sorted(singles, key=lambda r: (r["avg"] if not np.isnan(r["avg"]) else -1),
                     reverse=True)[:8]
    seed = []
    for r in singles:
        f = r["filters"]
        if f in filters and f not in seed:
            seed.append(f)
    # always include known helpful seeds
    for must in (["stretch_ok_short", "rsi65", "tod_am"] if side == "short"
                 else ["stretch_ok_long", "tod_am", "prior_down"]):
        if must in filters and must not in seed:
            seed.append(must)

    print(f"  triple seeds ({tf}|{base_name}): {seed[:10]}")
    for combo in itertools.combinations(seed[:10], 3):
        if len(set(combo) & {"tod_am", "tod_mid", "tod_pm"}) > 1:
            continue
        if len(set(combo) & {"gap_down", "gap_up"}) > 1:
            continue
        if len(set(combo) & {"prior_down", "prior_up"}) > 1:
            continue
        m = first & df[combo[0]].fillna(False) & df[combo[1]].fillna(False) & df[combo[2]].fillna(False)
        record(f"{tf}|{base_name}|{'+'.join(combo)}", m, "time")

    # also target-exit on top time-exit candidates later
    out = pd.DataFrame(rows)
    usable = out[(out["mode"] == "time") & (out["n"] >= min_n)].copy()
    if len(usable):
        usable["score"] = usable.apply(score, axis=1)
        usable = usable.sort_values("score", ascending=False)
        print(f"\n  TOP time-exit ({tf}|{base_name}, n>={min_n}):")
        for _, r in usable.head(12).iterrows():
            print(s.fmt_row(r))
        # re-test top 8 with target exit
        print(f"\n  Same top combos @ +0.15% target exit:")
        top_labels = usable.head(8)["label"].tolist()
        for lab in top_labels:
            # rebuild mask from filters string
            filt = lab.split("|", 2)[-1]
            if filt in ("alone",):
                m = first
            elif filt == ">>sma9":
                m = delayed
            else:
                parts = filt.split("+")
                m = first.copy()
                ok = True
                for p in parts:
                    if p not in df.columns:
                        ok = False
                        break
                    m = m & df[p].fillna(False)
                if not ok:
                    continue
            tr, r = backtest_target(df, m, side)
            r["label"] = lab + "@tgt"
            print(s.fmt_row(r) + (f"  tgt_hit={r['tgt_rate']:.0%}" if r["n"] else ""))
            rows.append({**r, "tf": tf, "base": base_name, "side": side,
                         "mode": "target", "filters": filt + "@tgt"})
    else:
        print(f"  (no combos with n>={min_n})")

    return pd.DataFrame(rows)


def walkforward_check(df, base_name, filt_parts, side, min_n=8):
    """Split sample in half chronologically; require both halves non-terrible."""
    base = s.base_mask(df, base_name)
    first = base & ~base.shift(1).fillna(False)
    m = first.copy()
    for p in filt_parts:
        if p in ("alone",):
            continue
        m = m & df[p].fillna(False)
    mid = df["ts"].quantile(0.5)
    a = df["ts"] <= mid
    b = df["ts"] > mid
    results = {}
    for name, sel in [("H1", a), ("H2", b)]:
        # mask only on selected rows: zero out others
        mm = m & sel
        _, r = s.backtest(df, mm, side, label=name, hold=HOLD)
        results[name] = r
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=MIN_N_DEFAULT)
    args = ap.parse_args()

    print("Building frames…")
    df5 = s.load_5m()
    daily = s.load_daily()
    frames = s.build_frames(df5, daily)
    for tf in ["5m", "15m", "30m", "1h"]:
        frames[tf] = enrich(frames[tf])

    targets = [
        ("5m", "bb_dn"),
        ("15m", "bb_dn"),
        ("5m", "bb_up"),
        ("15m", "bb_up"),
        ("30m", "bb_dn"),
        ("30m", "bb_up"),
    ]

    all_rows = []
    for tf, base in targets:
        print("\n" + "=" * 88)
        print(f"PHASE 3 — {tf}|{base}")
        print("=" * 88)
        all_rows.append(search_base(frames[tf], base, tf, min_n=args.min_n))

    out = pd.concat(all_rows, ignore_index=True)
    path = os.path.join(OUT, "signal_combo_phase3.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path} ({len(out)} rows)")

    # global leaderboard
    print("\n" + "=" * 88)
    print(f"GLOBAL LEADERBOARD — time exit, n>={args.min_n}, scored")
    print("=" * 88)
    board = out[(out["mode"] == "time") & (out["n"] >= args.min_n)].copy()
    board["score"] = board.apply(score, axis=1)
    board = board.sort_values("score", ascending=False)
    for _, r in board.head(20).iterrows():
        print(s.fmt_row(r))

    print("\n" + "=" * 88)
    print("WALK-FORWARD CHECK on top 10 (need both halves avg>=0 or WR>=50%)")
    print("=" * 88)
    survivors = []
    for _, r in board.head(15).iterrows():
        filt = str(r["filters"])
        if filt in ("alone", ">>sma9") or "@" in filt:
            parts = [] if filt == "alone" else []
            if filt == ">>sma9":
                print(f"  skip delayed for WF rebuild: {r['label']}")
                continue
        else:
            parts = filt.split("+")
        wf = walkforward_check(frames[r["tf"]], r["base"], parts, r["side"])
        h1, h2 = wf["H1"], wf["H2"]
        ok1 = h1["n"] >= 4 and (h1["avg"] >= 0 or h1["wr"] >= 0.5)
        ok2 = h2["n"] >= 4 and (h2["avg"] >= 0 or h2["wr"] >= 0.5)
        flag = "PASS" if (ok1 and ok2 and r["avg"] > 0) else "fail"
        print(f"  {flag:4} {r['label']:<48} "
              f"H1 n={h1['n']} WR={h1['wr']:.0%} avg={h1['avg']:+.3%} | "
              f"H2 n={h2['n']} WR={h2['wr']:.0%} avg={h2['avg']:+.3%}")
        if flag == "PASS":
            survivors.append(r)

    print("\n" + "=" * 88)
    print(f"SURVIVORS ({len(survivors)})")
    print("=" * 88)
    if survivors:
        surv = pd.DataFrame(survivors).drop_duplicates(subset=["label"])
        for _, r in surv.iterrows():
            print(s.fmt_row(r))
        surv.to_csv(os.path.join(OUT, "signal_combo_survivors.csv"), index=False)
        print(f"\nWrote signal_combo_survivors.csv")
    else:
        print("  none — loosen filters or gather more data")

    # short vs long summary
    print("\nBEST LONG / SHORT survivors:")
    for side in ("long", "short"):
        sub = [x for x in survivors if x["side"] == side]
        if not sub:
            print(f"  {side}: none")
            continue
        best = max(sub, key=score)
        print(f"  {side}: {s.fmt_row(best)}")


if __name__ == "__main__":
    main()
