#!/usr/bin/env python3
"""HTF-state confluence scan — 5m entries when higher TFs are in state[x].

For each 5m base recipe (L1/L2/L3/raw), test filters of the form:
  15m/30m/1h/1d/1w is in {bb_dn, bb_up, bb_mid, rsi_lo, rsi_hi, rsi_mid,
                          above_sma9, below_sma9, vwap_stretch_dn, ...}

Strategy (keeps nCk tractable + less overfit):
  1. Score EVERY single HTF-state filter alone
  2. Take top seeds per HTF (and globally)
  3. Enumerate combinations of size 2..K among seeds (default K=3)
  4. Gate: n>=min_n, WR>=min_wr, avg>=min_avg, improve vs base

No lookahead: HTF flags use the prior *completed* higher-TF bar (shift 1).

You do NOT need a larger chat context window for this — results land in CSV;
only survivors are printed.

Usage:
  python3 signal_htf_combo.py
  python3 signal_htf_combo.py --symbol QQQ --max-k 3 --min-n 12
  python3 signal_htf_combo.py --symbol SPY,QQQ,DIA --max-k 2
"""
from __future__ import annotations

import argparse
import itertools
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

HOLD = 5
HTF_RULES = {
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "1d": "1d",
    "1w": "1w",
}


def load_5m(sym):
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
    else:
        df = pd.read_csv(path)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    keep = (df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))
    return df.loc[keep].sort_values("ts").reset_index(drop=True)


def load_daily(sym):
    path = os.path.join(OUT, f"{sym}_daily.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        col = next(c for c in ("timestamps", "datetime", "ts", "date", "Date") if c in df.columns)
        df["ts"] = pd.to_datetime(df[col])
    else:
        import yfinance as yf
        raw = yf.download(sym, interval="1d", period="5y",
                          auto_adjust=True, progress=False).reset_index()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                           for c in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]
        ts = "date" if "date" in raw.columns else raw.columns[0]
        df = raw.rename(columns={ts: "ts"})
        df["ts"] = pd.to_datetime(df["ts"])
        df.to_csv(path, index=False)
    if df["ts"].dt.tz is None:
        df["ts"] = df["ts"].dt.tz_localize(NY, ambiguous="infer", nonexistent="shift_forward")
    else:
        df["ts"] = df["ts"].dt.tz_convert(NY)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"]).sort_values("ts").tail(800).reset_index(drop=True)


def htf_states(df, intraday=True):
    """Boolean state columns on a prepared OHLC frame."""
    df = s.prep(df, intraday=intraday)
    out = pd.DataFrame({"ts": df["ts"]})
    out["bb_dn"] = df["close"] <= df["bb_lo"]
    out["bb_up"] = df["close"] >= df["bb_up"]
    out["bb_mid"] = (~out["bb_dn"]) & (~out["bb_up"])
    out["rsi_lo"] = df["rsi"] <= 35
    out["rsi_hi"] = df["rsi"] >= 65
    out["rsi_mid"] = (~out["rsi_lo"]) & (~out["rsi_hi"])
    out["below_sma9"] = df["close"] < df["sma9"]
    out["above_sma9"] = df["close"] > df["sma9"]
    if intraday and "vwap_dist" in df.columns:
        out["vwap_dn"] = df["vwap_dist"] <= -0.002
        out["vwap_up"] = df["vwap_dist"] >= 0.002
    # green/red HTF candle
    out["candle_dn"] = df["close"] < df["open"]
    out["candle_up"] = df["close"] > df["open"]
    return out


def align_states(base_5m, htf_df, prefix):
    """merge_asof prior completed HTF states onto 5m bars."""
    h = htf_df.sort_values("ts").copy()
    # shift all state cols by 1 — only completed HTF bar
    state_cols = [c for c in h.columns if c != "ts"]
    for c in state_cols:
        h[c] = h[c].shift(1)
    left = pd.DataFrame({
        "ts_ns": pd.to_datetime(base_5m["ts"], utc=True).astype("int64").values,
    }, index=base_5m.index)
    right = pd.DataFrame({"ts_ns": pd.to_datetime(h["ts"], utc=True).astype("int64").values})
    for c in state_cols:
        right[f"{prefix}_{c}"] = h[c].values
    right = right.dropna(subset=[f"{prefix}_bb_dn"]).sort_values("ts_ns")
    m = pd.merge_asof(left.sort_values("ts_ns"), right, on="ts_ns", direction="backward")
    m = m.reindex(base_5m.index)
    out = base_5m.copy()
    for c in right.columns:
        if c == "ts_ns":
            continue
        out[c] = m[c].fillna(False).astype(bool).values
    return out


def build_panel(sym):
    print(f"Building panel {sym}…")
    df5 = load_5m(sym)
    daily = load_daily(sym)
    base = p3.enrich(s.prep(df5, intraday=True))

    # intraday HTFs from 5m
    for tf, rule in [("15m", "15min"), ("30m", "30min"), ("1h", "1h")]:
        raw = s.resample(df5, rule)
        st = htf_states(raw, intraday=True)
        base = align_states(base, st, tf)
        print(f"  aligned {tf}: {len(st)} bars")

    # daily
    st_d = htf_states(daily, intraday=False)
    base = align_states(base, st_d, "1d")
    print(f"  aligned 1d: {len(st_d)} bars")

    # weekly from daily
    w = daily.set_index("ts").resample("W-FRI", label="right", closed="right").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["close"]).reset_index()
    st_w = htf_states(w, intraday=False)
    base = align_states(base, st_w, "1w")
    print(f"  aligned 1w: {len(st_w)} bars")
    return base


def base_recipes(df):
    b = s.base_mask(df, "bb_dn")
    first = b & ~b.shift(1).fillna(False)

    def c(n):
        return df[n].fillna(False) if n in df.columns else pd.Series(False, index=df.index)

    return {
        "raw_bb_dn": first,
        "L1_prior_up": first & c("prior_up"),
        "L2_hvol": first & c("prior_up") & c("high_vol"),
        "L3_rsi35": first & c("prior_up") & c("rsi35"),
    }


def list_htf_flags(df):
    """All HTF state boolean columns on the panel."""
    prefs = ("15m_", "30m_", "1h_", "1d_", "1w_")
    return sorted(c for c in df.columns if c.startswith(prefs) and df[c].dtype == bool)


def summarize(tr_or_row):
    return tr_or_row


def fmt(r):
    if r["n"] == 0:
        return f"  {r['label']:<70} n=0"
    return (f"  {r['label']:<70} n={r['n']:>3} WR={r['wr']:.0%} "
            f"avg={r['avg']:+.3%} med={r['med']:+.3%} MFE={r['mfe_med']:+.3%}")


def score(r):
    if r["n"] < 1 or np.isnan(r.get("avg", np.nan)):
        return -1e9
    return r["avg"] * np.sqrt(r["n"]) * (0.5 + r["wr"])


def scan_symbol(sym, max_k=3, min_n=12, min_wr=0.60, min_avg=0.0005, seeds_per_tf=4):
    df = build_panel(sym)
    flags = list_htf_flags(df)
    print(f"  {len(flags)} HTF flags")
    recipes = base_recipes(df)
    rows = []

    for rname, base_mask in recipes.items():
        _, base_r = s.backtest(df, base_mask, "long", label=f"{sym}|5m|{rname}|alone", hold=HOLD)
        print(f"\n{sym} | {rname} alone: n={base_r['n']} WR={base_r['wr']:.0%} avg={base_r['avg']:+.3%}")
        rows.append({**base_r, "sym": sym, "recipe": rname, "combo": "alone",
                     "k": 0, "flags": "", "delta_avg": 0.0})

        # --- singles ---
        single_rows = []
        for f in flags:
            m = base_mask & df[f]
            _, r = s.backtest(df, m, "long", label=f"{sym}|{rname}+{f}", hold=HOLD)
            r.update(dict(sym=sym, recipe=rname, combo=f, k=1, flags=f,
                          delta_avg=(r["avg"] - base_r["avg"]) if r["n"] else np.nan,
                          base_avg=base_r["avg"], base_wr=base_r["wr"], base_n=base_r["n"]))
            single_rows.append(r)
            rows.append(r)

        singles = pd.DataFrame(single_rows)
        usable = singles[singles["n"] >= max(6, min_n // 2)].copy()
        if not len(usable):
            print("  (no single HTF filters with enough n)")
            continue
        usable["score"] = usable.apply(score, axis=1)
        # seeds: top overall + top per HTF prefix
        seeds = set()
        for _, r in usable.sort_values("score", ascending=False).head(12).iterrows():
            seeds.add(r["flags"])
        for pref in ("15m_", "30m_", "1h_", "1d_", "1w_"):
            sub = usable[usable["flags"].str.startswith(pref)].sort_values("score", ascending=False)
            for _, r in sub.head(seeds_per_tf).iterrows():
                seeds.add(r["flags"])
        seeds = sorted(seeds)
        print(f"  seeds ({len(seeds)}): " + ", ".join(seeds[:16]) + ("…" if len(seeds) > 16 else ""))

        print("  TOP singles:")
        for _, r in usable.sort_values("score", ascending=False).head(8).iterrows():
            print(fmt(r) + f"  Δavg={r['delta_avg']:+.3%}")

        # --- combos k=2..max_k among seeds ---
        for k in range(2, max_k + 1):
            # skip contradictory pairs within same HTF (bb_dn vs bb_up, etc.)
            combos_tested = 0
            for combo in itertools.combinations(seeds, k):
                # same-prefix exclusivity for bb_*/rsi_* families
                by_pref = {}
                bad = False
                for f in combo:
                    pref = f.split("_", 1)[0] + "_"
                    fam = f[len(pref):].rsplit("_", 1)[0] if False else None
                    # extract family: bb / rsi / sma9 / vwap / candle
                    rest = f[len(pref):]
                    family = rest.split("_")[0]  # bb, rsi, below/above -> handle
                    if rest.startswith("bb_"):
                        family = "bb"
                    elif rest.startswith("rsi_"):
                        family = "rsi"
                    elif rest.startswith("below_sma9") or rest.startswith("above_sma9"):
                        family = "sma9"
                    elif rest.startswith("vwap_"):
                        family = "vwap"
                    elif rest.startswith("candle_"):
                        family = "candle"
                    key = (pref, family)
                    if key in by_pref:
                        bad = True
                        break
                    by_pref[key] = f
                if bad:
                    continue
                m = base_mask.copy()
                for f in combo:
                    m = m & df[f]
                _, r = s.backtest(df, m, "long", label=f"{sym}|{rname}+{'+'.join(combo)}", hold=HOLD)
                combos_tested += 1
                r.update(dict(
                    sym=sym, recipe=rname, combo="+".join(combo), k=k,
                    flags="+".join(combo),
                    delta_avg=(r["avg"] - base_r["avg"]) if r["n"] else np.nan,
                    base_avg=base_r["avg"], base_wr=base_r["wr"], base_n=base_r["n"],
                ))
                rows.append(r)
            print(f"  k={k}: tested {combos_tested} combos")

    out = pd.DataFrame(rows)
    path = os.path.join(OUT, f"signal_htf_combo_{sym}.csv")
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")

    # survivors
    surv = out[
        (out["k"] >= 1)
        & (out["n"] >= min_n)
        & (out["wr"] >= min_wr)
        & (out["avg"] >= min_avg)
        & (out["delta_avg"] >= 0)  # at least as good as alone
    ].copy()
    if len(surv):
        surv["score"] = surv.apply(score, axis=1)
        surv = surv.sort_values("score", ascending=False)
    print("\n" + "=" * 88)
    print(f"SURVIVORS {sym} — n>={min_n} WR>={min_wr:.0%} avg>={min_avg:.3%} Δavg>=0")
    print("=" * 88)
    if not len(surv):
        print("  (none)")
    else:
        for _, r in surv.head(25).iterrows():
            print(fmt(r) + f"  Δ={r['delta_avg']:+.3%} k={r['k']}")
        surv.to_csv(os.path.join(OUT, f"signal_htf_survivors_{sym}.csv"), index=False)
    return out, surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY",
                    help="Comma-separated symbols, e.g. SPY,QQQ,DIA")
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--min-n", type=int, default=12)
    ap.add_argument("--min-wr", type=float, default=0.60)
    ap.add_argument("--min-avg", type=float, default=0.0005)
    args = ap.parse_args()

    print("NOTE: This is a compute/CSV job — a larger chat context window is NOT required.")
    print(f"max_k={args.max_k}  gate n>={args.min_n} WR>={args.min_wr:.0%} avg>={args.min_avg:.3%}")

    all_surv = []
    for sym in [x.strip().upper() for x in args.symbol.split(",")]:
        _, surv = scan_symbol(
            sym, max_k=args.max_k, min_n=args.min_n,
            min_wr=args.min_wr, min_avg=args.min_avg,
        )
        if len(surv):
            all_surv.append(surv)

    if all_surv:
        big = pd.concat(all_surv, ignore_index=True).sort_values("score", ascending=False)
        path = os.path.join(OUT, "signal_htf_survivors_all.csv")
        big.to_csv(path, index=False)
        print("\n" + "=" * 88)
        print("GLOBAL TOP HTF COMBOS")
        print("=" * 88)
        for _, r in big.head(30).iterrows():
            print(fmt(r) + f"  Δ={r['delta_avg']:+.3%}")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
