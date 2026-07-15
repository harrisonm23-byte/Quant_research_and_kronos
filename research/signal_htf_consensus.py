#!/usr/bin/env python3
"""Cross-symbol consensus over HTF combo CSVs.

Reads signal_htf_combo_{SYM}.csv (from signal_htf_combo.py) and ranks
single flags / pairs by how many symbols they improve on.

Prefers k=1 and k=2 — k=3 rows are mostly noise on ~60d samples.

Usage:
  python3 signal_htf_consensus.py
  python3 signal_htf_consensus.py --symbols SPY,QQQ,DIA,IWM --min-syms 2
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
RECIPES = ["raw_bb_dn", "L1_prior_up", "L2_hvol", "L3_rsi35"]


def load_combos(symbols):
    frames = {}
    for sym in symbols:
        path = os.path.join(OUT, f"signal_htf_combo_{sym}.csv")
        if not os.path.exists(path):
            print(f"  missing {path} — skip {sym}")
            continue
        frames[sym] = pd.read_csv(path)
        print(f"  loaded {sym}: {len(frames[sym])} rows")
    return frames


def norm_combo(flags: str) -> str:
    parts = [p.strip() for p in str(flags).split("+") if p.strip() and p != "alone"]
    return "+".join(sorted(parts))


def score_row(r):
    if r["n"] < 1 or np.isnan(r.get("avg", np.nan)):
        return -1e9
    return float(r["avg"]) * np.sqrt(float(r["n"])) * (0.5 + float(r["wr"]))


def consensus_table(frames, k, min_n, min_wr, min_delta, min_syms):
    """Return DataFrame of combos recurring across symbols."""
    # key = (recipe, normalized_combo) -> list of per-symbol rows
    bucket = defaultdict(list)
    for sym, df in frames.items():
        sub = df[
            (df["k"] == k)
            & (df["n"] >= min_n)
            & (df["wr"] >= min_wr)
            & (df["delta_avg"] >= min_delta)
        ].copy()
        for _, r in sub.iterrows():
            key = (r["recipe"], norm_combo(r["flags"]))
            bucket[key].append({
                "sym": sym,
                "n": int(r["n"]),
                "wr": float(r["wr"]),
                "avg": float(r["avg"]),
                "delta_avg": float(r["delta_avg"]),
                "score": score_row(r),
            })

    rows = []
    n_sym_total = len(frames)
    for (recipe, combo), items in bucket.items():
        syms = sorted({x["sym"] for x in items})
        if len(syms) < min_syms:
            continue
        rows.append({
            "recipe": recipe,
            "combo": combo,
            "k": k,
            "n_syms": len(syms),
            "syms": ",".join(syms),
            "coverage": f"{len(syms)}/{n_sym_total}",
            "mean_n": np.mean([x["n"] for x in items]),
            "mean_wr": np.mean([x["wr"] for x in items]),
            "mean_avg": np.mean([x["avg"] for x in items]),
            "mean_delta": np.mean([x["delta_avg"] for x in items]),
            "min_n": min(x["n"] for x in items),
            "min_wr": min(x["wr"] for x in items),
            "min_avg": min(x["avg"] for x in items),
            "sum_score": sum(x["score"] for x in items),
            "detail": " | ".join(
                f"{x['sym']} n={x['n']} WR={x['wr']:.0%} avg={x['avg']:+.3%} Δ={x['delta_avg']:+.3%}"
                for x in sorted(items, key=lambda z: z["sym"])
            ),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["n_syms", "sum_score", "mean_delta"], ascending=[False, False, False]
        )
    return out


def motif_table(singles_df):
    """Which individual HTF states appear most in multi-symbol singles."""
    if not len(singles_df):
        return pd.DataFrame()
    # explode is already one flag per row for k=1
    g = singles_df.groupby(["recipe", "combo"]).agg(
        n_syms=("n_syms", "max"),
        mean_delta=("mean_delta", "mean"),
        mean_wr=("mean_wr", "mean"),
        mean_n=("mean_n", "mean"),
        sum_score=("sum_score", "sum"),
        syms=("syms", "first"),
    ).reset_index()
    return g.sort_values(["n_syms", "sum_score"], ascending=[False, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY,QQQ,DIA,IWM")
    ap.add_argument("--min-syms", type=int, default=2)
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--min-wr", type=float, default=0.55)
    ap.add_argument("--min-delta", type=float, default=0.0)
    args = ap.parse_args()

    symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    print("HTF consensus — no larger chat context needed; this reads CSVs.")
    frames = load_combos(symbols)
    if not frames:
        print("No combo CSVs found. Run signal_htf_combo.py first.")
        return

    singles = consensus_table(
        frames, k=1, min_n=args.min_n, min_wr=args.min_wr,
        min_delta=args.min_delta, min_syms=args.min_syms,
    )
    pairs = consensus_table(
        frames, k=2, min_n=args.min_n, min_wr=max(args.min_wr, 0.60),
        min_delta=args.min_delta, min_syms=args.min_syms,
    )

    sp = os.path.join(OUT, "signal_htf_consensus_singles.csv")
    pp = os.path.join(OUT, "signal_htf_consensus_pairs.csv")
    singles.to_csv(sp, index=False)
    pairs.to_csv(pp, index=False)
    print(f"\nWrote {sp} ({len(singles)} rows)")
    print(f"Wrote {pp} ({len(pairs)} rows)")

    # Promotion shortlist: 3+ symbols OR (2+ with strong Δ and n)
    n_need = min(3, len(frames))
    promo = []
    for df, kind in [(singles, "single"), (pairs, "pair")]:
        if not len(df):
            continue
        for _, r in df.iterrows():
            strong = (
                r["n_syms"] >= n_need
                or (r["n_syms"] >= 2 and r["mean_delta"] >= 0.0003 and r["min_n"] >= 12)
            )
            if not strong:
                continue
            if r["recipe"] == "raw_bb_dn" and r["mean_wr"] < 0.60:
                continue  # raw needs higher bar
            promo.append({**r.to_dict(), "kind": kind})
    promo_df = pd.DataFrame(promo)
    if len(promo_df):
        promo_df = promo_df.sort_values(
            ["n_syms", "sum_score"], ascending=[False, False]
        )
    promo_path = os.path.join(OUT, "signal_htf_consensus_promote.csv")
    promo_df.to_csv(promo_path, index=False)
    print(f"Wrote {promo_path} ({len(promo_df)} candidates)")

    print("\n" + "=" * 88)
    print(f"TOP CONSENSUS SINGLES (k=1, ≥{args.min_syms} symbols)")
    print("=" * 88)
    for recipe in RECIPES:
        sub = singles[singles["recipe"] == recipe].head(10)
        if not len(sub):
            continue
        print(f"\n--- {recipe} ---")
        for _, r in sub.iterrows():
            print(
                f"  [{r['coverage']}] {r['combo']:<28} "
                f"meanΔ={r['mean_delta']:+.3%} meanWR={r['mean_wr']:.0%} "
                f"meanN={r['mean_n']:.0f}"
            )
            print(f"           {r['detail']}")

    print("\n" + "=" * 88)
    print(f"TOP CONSENSUS PAIRS (k=2, ≥{args.min_syms} symbols)")
    print("=" * 88)
    for recipe in RECIPES:
        sub = pairs[pairs["recipe"] == recipe].head(10)
        if not len(sub):
            continue
        print(f"\n--- {recipe} ---")
        for _, r in sub.iterrows():
            print(
                f"  [{r['coverage']}] {r['combo']:<50} "
                f"meanΔ={r['mean_delta']:+.3%} meanWR={r['mean_wr']:.0%}"
            )

    print("\n" + "=" * 88)
    print("PROMOTION SHORTLIST (for keepers)")
    print("=" * 88)
    if not len(promo_df):
        print("  (none)")
    else:
        for _, r in promo_df.head(25).iterrows():
            print(
                f"  [{r['coverage']}] {r['kind']:<6} {r['recipe']}+{r['combo']:<40} "
                f"Δ={r['mean_delta']:+.3%} WR={r['mean_wr']:.0%} n~{r['mean_n']:.0f}"
            )


if __name__ == "__main__":
    main()
