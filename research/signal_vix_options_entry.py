#!/usr/bin/env python3
"""What happens if we buy options on L1/L2/L3 signals only when VIX is low?

Joins the prior session's VIX close (no lookahead) to every trade produced by
the execution-correct exit engine, then reports per-bucket:
  * underlying trade stats (WR / avg), and
  * modeled ATM-call P&L where the IV *scales with the entry VIX level* —
    so cheap-premium-at-low-VIX is priced in, not assumed away.

Buckets are reported two ways:
  * absolute VIX level (<13, 13-15, 15-18, 18-22, 22-30, >=30)
  * trailing 252-session VIX percentile (0-20 = "specific low", ..., 80-100)

Pricing: Black-Scholes, 2 DTE ATM call, IV = (VIX/100) * symbol multiplier
(SPY 1.0, QQQ 1.2, TQQQ 3.0 — crude beta scaling), spot-vol beta at exit,
2% premium haircut round trip. Modeled only; no real chains.

Usage:
  python3 signal_vix_options_entry.py \
    --trades signal_exit_mechanics_SPY_historical_2019_2021_trades.csv \
    --symbol SPY --tag historical_2019_2021
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
R = 0.04
COST = 0.02
DTE = 2
IV_SYMBOL_MULT = {"SPY": 1.0, "QQQ": 1.2, "TQQQ": 3.0}

LEVEL_BUCKETS = [
    ("<13", 0, 13), ("13-15", 13, 15), ("15-18", 15, 18),
    ("18-22", 18, 22), ("22-30", 22, 30), (">=30", 30, 999),
]
PCT_BUCKETS = [
    ("pct 0-20 (low)", 0.0, 0.20), ("pct 20-40", 0.20, 0.40),
    ("pct 40-60", 0.40, 0.60), ("pct 60-80", 0.60, 0.80),
    ("pct 80-100 (high)", 0.80, 1.01),
]


def fetch_full_vix(path):
    import yfinance as yf
    raw = yf.download("^VIX", interval="1d", period="max",
                      auto_adjust=False, progress=False).reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]
    ts = "date" if "date" in raw.columns else raw.columns[0]
    df = raw.rename(columns={ts: "ts"})[["ts", "open", "high", "low", "close"]]
    df.to_csv(path, index=False)
    return path


def load_vix():
    full = os.path.join(OUT, "VIX_daily_full.csv")
    if not os.path.exists(full):
        try:
            fetch_full_vix(full)
        except Exception as e:
            print(f"full VIX fetch failed ({e}); falling back to local caches")
    for path in (full, "/tmp/VIX_daily_full.csv", os.path.join(OUT, "VIX_daily.csv")):
        if os.path.exists(path):
            v = pd.read_csv(path)
            col = next(c for c in v.columns if c.lower() in
                       ("ts", "date", "datetime", "timestamps"))
            v["date"] = pd.to_datetime(v[col], format="mixed", utc=True).dt.date
            v = v[["date", "close"]].rename(columns={"close": "vix"})
            v = v.dropna().sort_values("date").reset_index(drop=True)
            # prior close only — a trade on day D sees VIX through D-1
            v["vix_prior"] = v["vix"].shift(1)
            v["vix_pct_252"] = (
                v["vix_prior"].rolling(252, min_periods=100)
                .apply(lambda w: (w <= w.iloc[-1]).mean())
            )
            print(f"VIX source {path}: {v['date'].min()} -> {v['date'].max()}")
            return v
    raise FileNotFoundError("no VIX daily file")


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_call(spot, strike, years, iv):
    if years <= 0:
        return max(spot - strike, 0.0)
    iv = max(iv, 0.01)
    d1 = (math.log(spot / strike) + (R + iv * iv / 2) * years) / (iv * math.sqrt(years))
    d2 = d1 - iv * math.sqrt(years)
    return spot * normal_cdf(d1) - strike * math.exp(-R * years) * normal_cdf(d2)


def option_pnl(row, iv_mult):
    """ATM 2-DTE call bought at entry, sold at the strategy exit."""
    s0, sx = row["entry"], row["exit"]
    iv0 = (row["vix_prior"] / 100.0) * iv_mult
    strike = s0
    elapsed_days = max(row["bars"], 1) / 78
    t0 = DTE / 252
    tx = max(DTE - elapsed_days, 0.02) / 252
    ret = sx / s0 - 1
    ive = iv0 * min(1.4, max(0.6, 1 - 3 * ret))
    p0 = bs_call(s0, strike, t0, iv0) * (1 + COST / 2)
    px = bs_call(sx, strike, tx, ive) * (1 - COST / 2)
    if p0 < 0.01:
        return np.nan
    return px / p0 - 1


def bucket_stats(df, col, buckets):
    rows = []
    for name, lo, hi in buckets:
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        if not len(sub):
            rows.append(dict(bucket=name, n=0))
            continue
        opt = sub["opt_ret"].dropna()
        rows.append(dict(
            bucket=name, n=len(sub),
            und_wr=(sub["ret"] > 0).mean(), und_avg=sub["ret"].mean(),
            opt_n=len(opt),
            opt_wr=(opt > 0).mean() if len(opt) else np.nan,
            opt_avg=opt.mean() if len(opt) else np.nan,
            opt_med=opt.median() if len(opt) else np.nan,
            opt_p05=opt.quantile(0.05) if len(opt) else np.nan,
            vix_mean=sub["vix_prior"].mean(),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", required=True,
                    help="exit-mechanics trades CSV (relative to research/)")
    ap.add_argument("--symbol", required=True, choices=list(IV_SYMBOL_MULT))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--setups", default="L1_first,L2_first,L3_first")
    ap.add_argument("--mechanics", default="fixed_24,fixed_eod")
    args = ap.parse_args()

    path = args.trades if os.path.isabs(args.trades) else os.path.join(OUT, args.trades)
    trades = pd.read_csv(path)
    trades = trades[
        trades["setup"].isin(args.setups.split(","))
        & trades["mechanic"].isin(args.mechanics.split(","))
    ].copy()
    trades["date"] = pd.to_datetime(
        trades["signal_ts"], format="mixed", utc=True
    ).dt.tz_convert("America/New_York").dt.date

    vix = load_vix()
    trades = trades.merge(
        vix[["date", "vix_prior", "vix_pct_252"]], on="date", how="left"
    )
    missing = trades["vix_prior"].isna().sum()
    if missing:
        print(f"NOTE: dropping {missing} trades with no prior VIX (weekend/missing)")
    trades = trades.dropna(subset=["vix_prior"]).reset_index(drop=True)

    iv_mult = IV_SYMBOL_MULT[args.symbol]
    trades["opt_ret"] = trades.apply(lambda r: option_pnl(r, iv_mult), axis=1)

    all_out = []
    for (setup, mech), grp in trades.groupby(["setup", "mechanic"]):
        for col, buckets, kind in [
            ("vix_prior", LEVEL_BUCKETS, "level"),
            ("vix_pct_252", PCT_BUCKETS, "pct252"),
        ]:
            tab = bucket_stats(grp, col, buckets)
            tab.insert(0, "kind", kind)
            tab.insert(0, "mechanic", mech)
            tab.insert(0, "setup", setup)
            all_out.append(tab)

    out = pd.concat(all_out, ignore_index=True)
    out_path = os.path.join(
        OUT, f"signal_vix_options_entry_{args.symbol}_{args.tag}.csv"
    )
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)\n")
    print("MODELED ONLY: BS + VIX-proportional IV; no real chains.\n")

    for (setup, mech), grp in out.groupby(["setup", "mechanic"]):
        print(f"=== {args.symbol} {setup} {mech} ===")
        for kind in ("level", "pct252"):
            sub = grp[grp["kind"] == kind]
            print(f"  by VIX {kind}:")
            for _, r in sub.iterrows():
                if not r["n"]:
                    print(f"    {r['bucket']:<18} n=0")
                    continue
                print(
                    f"    {r['bucket']:<18} n={int(r['n']):>3} "
                    f"und WR={r['und_wr']:.0%} avg={r['und_avg']:+.3%} | "
                    f"opt WR={r['opt_wr']:.0%} avg={r['opt_avg']:+.1%} "
                    f"med={r['opt_med']:+.1%} p05={r['opt_p05']:+.1%} "
                    f"(VIX~{r['vix_mean']:.1f})"
                )
        print()


if __name__ == "__main__":
    main()
