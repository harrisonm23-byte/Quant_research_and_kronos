#!/usr/bin/env python3
"""Does VIX level/change affect BB-fade keeper returns?

Aligns ^VIX (daily + 5m when available) onto SPY signal bars with no lookahead
(prior completed VIX bar via merge_asof).

Tests:
  1. Return buckets by VIX level, 1d change, rising/falling, percentile
  2. VIX filters as add-ons to each keeper (low/mid/high, rising/falling)
  3. Interaction: longs vs shorts × VIX regime

Usage:
  python3 signal_vix_study.py
  python3 signal_vix_study.py --min-n 8
"""
from __future__ import annotations

import argparse
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
sys.path.insert(0, OUT)

import signal_combo_scan as s
import signal_combo_phase3 as p3
import signal_keepers as sk

HOLD = 5
VIX_DAILY = os.path.join(OUT, "VIX_daily.csv")
VIX_5M = os.path.join(OUT, "VIX_5m_yf.csv")


def _flatten_yf(raw):
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower()
                       for c in raw.columns]
    else:
        raw.columns = [str(c).lower() for c in raw.columns]
    return raw


def fetch_vix_daily():
    if os.path.exists(VIX_DAILY):
        df = pd.read_csv(VIX_DAILY)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        print(f"Loaded {VIX_DAILY}: {len(df)} days")
        return df
    import yfinance as yf
    raw = yf.download("^VIX", interval="1d", period="5y",
                      auto_adjust=True, progress=False).reset_index()
    raw = _flatten_yf(raw)
    ts_col = "date" if "date" in raw.columns else raw.columns[0]
    df = raw.rename(columns={ts_col: "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)
    df.to_csv(VIX_DAILY, index=False)
    print(f"Fetched daily VIX: {len(df)} days -> {VIX_DAILY}")
    return df


def fetch_vix_5m():
    if os.path.exists(VIX_5M):
        df = pd.read_csv(VIX_5M)
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
        print(f"Loaded {VIX_5M}: {len(df)} bars")
        return df
    import yfinance as yf
    raw = yf.download("^VIX", interval="5m", period="60d",
                      auto_adjust=True, progress=False).reset_index()
    raw = _flatten_yf(raw)
    ts_col = "datetime" if "datetime" in raw.columns else raw.columns[0]
    df = raw.rename(columns={ts_col: "ts"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(NY)
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("ts").reset_index(drop=True)
    df.to_csv(VIX_5M, index=False)
    print(f"Fetched 5m VIX: {len(df)} bars -> {VIX_5M}")
    return df


def prep_vix_daily(v):
    v = v.copy()
    v["vix"] = v["close"]
    v["vix_chg1d"] = v["vix"].pct_change()
    v["vix_chg_pts"] = v["vix"].diff()
    v["vix_ma10"] = v["vix"].rolling(10).mean()
    v["vix_ma20"] = v["vix"].rolling(20).mean()
    v["vix_above_ma10"] = v["vix"] > v["vix_ma10"]
    v["vix_rising"] = v["vix_chg_pts"] > 0
    v["vix_pctile"] = v["vix"].rolling(60).rank(pct=True)
    # regime by level
    v["vix_regime"] = pd.cut(
        v["vix"],
        bins=[-np.inf, 15, 18, 22, 28, np.inf],
        labels=["lt15", "15_18", "18_22", "22_28", "gt28"],
    )
    return v


def prep_vix_5m(v5):
    v5 = v5.copy()
    v5["vix5"] = v5["close"]
    v5["vix5_chg"] = v5["vix5"].pct_change()
    v5["vix5_rising"] = v5["vix5"].diff() > 0
    # session VWAP-ish not needed; use rolling z
    v5["vix5_z"] = (v5["vix5"] - v5["vix5"].rolling(78).mean()) / v5["vix5"].rolling(78).std()
    return v5


def align_vix(df, vix_d, vix_5m=None):
    """Attach prior-day VIX features + optional prior 5m VIX to SPY bars."""
    # daily: use prior completed daily bar (shift 1 on daily, then asof)
    vd = vix_d[["ts", "vix", "vix_chg1d", "vix_chg_pts", "vix_above_ma10",
                "vix_rising", "vix_pctile", "vix_regime", "vix_ma10", "vix_ma20"]].copy()
    vd = vd.sort_values("ts")
    for c in ["vix", "vix_chg1d", "vix_chg_pts", "vix_above_ma10", "vix_rising",
              "vix_pctile", "vix_regime", "vix_ma10", "vix_ma20"]:
        vd[c] = vd[c].shift(1)  # no lookahead: yesterday's close for today's signals

    left = pd.DataFrame({
        "ts_ns": pd.to_datetime(df["ts"], utc=True).astype("int64").values,
    }, index=df.index)
    right = pd.DataFrame({
        "ts_ns": pd.to_datetime(vd["ts"], utc=True).astype("int64").values,
        "vix": vd["vix"].values,
        "vix_chg1d": vd["vix_chg1d"].values,
        "vix_chg_pts": vd["vix_chg_pts"].values,
        "vix_above_ma10": vd["vix_above_ma10"].values,
        "vix_rising": vd["vix_rising"].values,
        "vix_pctile": vd["vix_pctile"].values,
        "vix_regime": vd["vix_regime"].astype(str).values,
        "vix_ma10": vd["vix_ma10"].values,
        "vix_ma20": vd["vix_ma20"].values,
    }).dropna(subset=["vix"]).sort_values("ts_ns")

    m = pd.merge_asof(left.sort_values("ts_ns"), right, on="ts_ns", direction="backward")
    m = m.reindex(df.index)
    out = df.copy()
    for c in ["vix", "vix_chg1d", "vix_chg_pts", "vix_above_ma10", "vix_rising",
              "vix_pctile", "vix_regime", "vix_ma10", "vix_ma20"]:
        out[c] = m[c].values

    # derived filters (boolean)
    out["vix_low"] = out["vix"] < 15
    out["vix_mid"] = (out["vix"] >= 15) & (out["vix"] < 22)
    out["vix_high"] = out["vix"] >= 22
    out["vix_very_high"] = out["vix"] >= 28
    out["vix_up_day"] = out["vix_chg_pts"] >= 0.5
    out["vix_dn_day"] = out["vix_chg_pts"] <= -0.5
    out["vix_calm_chg"] = out["vix_chg_pts"].abs() < 0.5
    out["vix_pct_hi"] = out["vix_pctile"] >= 0.70
    out["vix_pct_lo"] = out["vix_pctile"] <= 0.30

    if vix_5m is not None and len(vix_5m):
        v5 = vix_5m[["ts", "vix5", "vix5_rising", "vix5_z"]].copy().sort_values("ts")
        for c in ["vix5", "vix5_rising", "vix5_z"]:
            v5[c] = v5[c].shift(1)
        r5 = pd.DataFrame({
            "ts_ns": pd.to_datetime(v5["ts"], utc=True).astype("int64").values,
            "vix5": v5["vix5"].values,
            "vix5_rising": v5["vix5_rising"].values,
            "vix5_z": v5["vix5_z"].values,
        }).dropna(subset=["vix5"]).sort_values("ts_ns")
        m5 = pd.merge_asof(left.sort_values("ts_ns"), r5, on="ts_ns", direction="backward")
        m5 = m5.reindex(df.index)
        out["vix5"] = m5["vix5"].values
        out["vix5_rising"] = m5["vix5_rising"].values
        out["vix5_z"] = m5["vix5_z"].values
        out["vix5_spike"] = out["vix5_z"] >= 1.0
        out["vix5_crush"] = out["vix5_z"] <= -1.0
    return out


def trade_frame(df, mask, side, hold=HOLD):
    """Return per-trade dataframe with VIX context at signal bar."""
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    m = mask.fillna(False).values
    n = len(df)
    rows, last = [], -10**9
    for i in range(30, n - hold - 1):
        if not m[i] or i - last < 5:
            continue
        entry = o[i + 1] * (1 + s.SLIP if side == "long" else 1 - s.SLIP)
        end = min(i + 1 + hold, n)
        path_h, path_l, path_c = h[i + 1:end], l[i + 1:end], c[i + 1:end]
        if len(path_c) == 0:
            continue
        exit_px = path_c[-1]
        if side == "short":
            ret = entry / exit_px - 1.0
            mfe = entry / path_l.min() - 1.0
        else:
            ret = exit_px / entry - 1.0
            mfe = path_h.max() / entry - 1.0
        r = df.iloc[i]
        rows.append(dict(
            ts=r["ts"], day=r["day"], side=side, ret=ret, mfe=mfe,
            vix=r.get("vix", np.nan),
            vix_chg_pts=r.get("vix_chg_pts", np.nan),
            vix_regime=r.get("vix_regime", ""),
            vix_rising=r.get("vix_rising", np.nan),
            vix_pctile=r.get("vix_pctile", np.nan),
            vix5=r.get("vix5", np.nan),
            vix5_rising=r.get("vix5_rising", np.nan),
            vix5_z=r.get("vix5_z", np.nan),
        ))
        last = i
    return pd.DataFrame(rows)


def summarize(tr, label=""):
    if not len(tr):
        return dict(label=label, n=0, wr=np.nan, avg=np.nan, med=np.nan,
                    mfe_med=np.nan, sum=0.0)
    return dict(
        label=label, n=len(tr),
        wr=float((tr["ret"] > 0).mean()),
        avg=float(tr["ret"].mean()),
        med=float(tr["ret"].median()),
        mfe_med=float(tr["mfe"].median()),
        sum=float(tr["ret"].sum()),
        vix_med=float(tr["vix"].median()) if tr["vix"].notna().any() else np.nan,
    )


def fmt(r):
    if r["n"] == 0:
        return f"  {r['label']:<48} n=0"
    return (f"  {r['label']:<48} n={r['n']:>3}  WR={r['wr']:.0%}  "
            f"avg={r['avg']:+.3%}  med={r['med']:+.3%}  "
            f"MFE={r['mfe_med']:+.3%}  sum={r['sum']:+.2%}")


KEEPERS = [
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

VIX_FILTERS = [
    "vix_low", "vix_mid", "vix_high", "vix_very_high",
    "vix_rising", "vix_up_day", "vix_dn_day", "vix_calm_chg",
    "vix_above_ma10", "vix_pct_hi", "vix_pct_lo",
    "vix5_rising", "vix5_spike", "vix5_crush",
]


def bucket_report(tr, keeper, min_n):
    rows = []
    print(f"\n-- {keeper} buckets (all n={len(tr)}) --")
    base = summarize(tr, f"{keeper}|ALL")
    print(fmt(base))
    rows.append({**base, "keeper": keeper, "kind": "all"})

    if not len(tr):
        return rows

    # level regimes
    for reg, g in tr.groupby("vix_regime", dropna=False):
        r = summarize(g, f"{keeper}|regime={reg}")
        if r["n"] >= min_n:
            print(fmt(r))
        rows.append({**r, "keeper": keeper, "kind": "regime", "bucket": str(reg)})

    # rising / falling (prior day)
    for flag, lab in [(True, "vix_rising"), (False, "vix_falling")]:
        g = tr[tr["vix_rising"] == flag]
        r = summarize(g, f"{keeper}|{lab}")
        if r["n"] >= min_n:
            print(fmt(r))
        rows.append({**r, "keeper": keeper, "kind": "rising", "bucket": lab})

    # chg buckets
    tr = tr.copy()
    tr["chg_bucket"] = pd.cut(
        tr["vix_chg_pts"],
        bins=[-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf],
        labels=["crush", "soft_dn", "flat", "soft_up", "spike"],
    )
    for b, g in tr.groupby("chg_bucket", observed=True):
        r = summarize(g, f"{keeper}|chg={b}")
        if r["n"] >= min_n:
            print(fmt(r))
        rows.append({**r, "keeper": keeper, "kind": "chg", "bucket": str(b)})

    # percentile
    tr["pct_bucket"] = pd.cut(
        tr["vix_pctile"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["pct_lo", "pct_mid", "pct_hi"],
    )
    for b, g in tr.groupby("pct_bucket", observed=True):
        r = summarize(g, f"{keeper}|{b}")
        if r["n"] >= min_n:
            print(fmt(r))
        rows.append({**r, "keeper": keeper, "kind": "pctile", "bucket": str(b)})

    # intraday VIX z if present
    if tr["vix5_z"].notna().any():
        tr["z_bucket"] = pd.cut(
            tr["vix5_z"],
            bins=[-np.inf, -0.5, 0.5, np.inf],
            labels=["z_neg", "z_flat", "z_pos"],
        )
        for b, g in tr.groupby("z_bucket", observed=True):
            r = summarize(g, f"{keeper}|{b}")
            if r["n"] >= min_n:
                print(fmt(r))
            rows.append({**r, "keeper": keeper, "kind": "vix5z", "bucket": str(b)})
    return rows


def filter_search(df, mask, side, keeper, min_n):
    """Apply VIX boolean filters on top of keeper mask."""
    rows = []
    print(f"\n-- {keeper} + VIX filters --")
    _, base_r = s.backtest(df, mask, side, label=f"{keeper}|alone", hold=HOLD)
    print(fmt({**base_r, "label": f"{keeper}|alone", "mfe_med": base_r["mfe_med"]}))
    rows.append({**base_r, "keeper": keeper, "filter": "alone", "kind": "filter"})

    for f in VIX_FILTERS:
        if f not in df.columns:
            continue
        m = mask & df[f].fillna(False)
        tr = trade_frame(df, m, side)
        r = summarize(tr, f"{keeper}+{f}")
        if r["n"] >= min_n:
            print(fmt(r))
        rows.append({**r, "keeper": keeper, "filter": f, "kind": "filter"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=8)
    args = ap.parse_args()

    print("Loading SPY + VIX…")
    df5 = s.load_5m()
    daily = s.load_daily()
    frames = s.build_frames(df5, daily)
    for tf in ["5m", "15m", "30m", "1h"]:
        frames[tf] = p3.enrich(frames[tf])

    vix_d = prep_vix_daily(fetch_vix_daily())
    try:
        vix_5m = prep_vix_5m(fetch_vix_5m())
    except Exception as e:
        print(f"5m VIX unavailable ({e}); using daily only")
        vix_5m = None

    for tf in ["5m", "15m"]:
        frames[tf] = align_vix(frames[tf], vix_d, vix_5m)
        print(f"  {tf} VIX aligned: med={frames[tf]['vix'].median():.1f}  "
              f"range {frames[tf]['vix'].min():.1f}-{frames[tf]['vix'].max():.1f}")

    all_bucket, all_filt, all_trades = [], [], []

    print("\n" + "=" * 88)
    print("VIX CONDITIONING — keepers")
    print("=" * 88)

    for tf, name in KEEPERS:
        df = frames[tf]
        mask, side = sk.masks(df, name)
        tr = trade_frame(df, mask, side)
        if len(tr):
            tr["keeper"] = name
            all_trades.append(tr)
        all_bucket.extend(bucket_report(tr, name, args.min_n))
        all_filt.extend(filter_search(df, mask, side, name, args.min_n))

    # Base bb_dn / bb_up alone for broader sample
    print("\n" + "=" * 88)
    print("VIX CONDITIONING — raw BB bases (larger n)")
    print("=" * 88)
    for tf, base in [("5m", "bb_dn"), ("5m", "bb_up"), ("15m", "bb_dn"), ("15m", "bb_up")]:
        df = frames[tf]
        b = s.base_mask(df, base)
        first = b & ~b.shift(1).fillna(False)
        side = s.side_of(base)
        name = f"{tf}|{base}"
        tr = trade_frame(df, first, side)
        if len(tr):
            tr["keeper"] = name
            all_trades.append(tr)
        all_bucket.extend(bucket_report(tr, name, args.min_n))
        all_filt.extend(filter_search(df, first, side, name, args.min_n))

    buck = pd.DataFrame(all_bucket)
    filt = pd.DataFrame(all_filt)
    buck.to_csv(os.path.join(OUT, "signal_vix_buckets.csv"), index=False)
    filt.to_csv(os.path.join(OUT, "signal_vix_filters.csv"), index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(
            os.path.join(OUT, "signal_vix_trades.csv"), index=False)

    # Leaderboard: filters that improve avg vs alone for same keeper
    print("\n" + "=" * 88)
    print(f"BEST VIX FILTERS (n>={args.min_n}, improve avg vs alone)")
    print("=" * 88)
    improvements = []
    for keeper, g in filt.groupby("keeper"):
        alone = g[g["filter"] == "alone"]
        if not len(alone) or alone.iloc[0]["n"] < 1:
            continue
        base_avg = alone.iloc[0]["avg"]
        base_wr = alone.iloc[0]["wr"]
        for _, r in g[g["filter"] != "alone"].iterrows():
            if r["n"] < args.min_n or np.isnan(r["avg"]):
                continue
            if r["avg"] > base_avg + 0.0001:  # at least +1bp improvement
                improvements.append({**r, "base_avg": base_avg, "base_wr": base_wr,
                                     "delta_avg": r["avg"] - base_avg})
    if improvements:
        imp = pd.DataFrame(improvements).sort_values("delta_avg", ascending=False)
        for _, r in imp.head(25).iterrows():
            print(f"  {r['label']:<48} n={r['n']:>3} WR={r['wr']:.0%} "
                  f"avg={r['avg']:+.3%} (Δ {r['delta_avg']:+.3%} vs alone)")
        imp.to_csv(os.path.join(OUT, "signal_vix_improvements.csv"), index=False)
    else:
        print("  (none clear)")

    # Interaction summary: long keepers prefer which VIX?
    print("\n" + "=" * 88)
    print("INTERACTION SUMMARY — L1/L2/L3 by VIX regime")
    print("=" * 88)
    for name in ["L1_5m_bbdn_prior_up", "L2_5m_bbdn_prior_up_hvol", "L3_5m_bbdn_prior_up_rsi35",
                 "S1_15m_bbup_vwap_rsi65", "S2_15m_bbup_gap_up_stretch025"]:
        sub = buck[(buck["keeper"] == name) & (buck["kind"] == "regime") & (buck["n"] >= args.min_n)]
        if not len(sub):
            # also show rising
            sub = buck[(buck["keeper"] == name) & (buck["kind"].isin(["rising", "chg"]))
                       & (buck["n"] >= args.min_n)]
        for _, r in sub.iterrows():
            print(fmt(r))

    print("\nWrote signal_vix_buckets.csv / signal_vix_filters.csv / "
          "signal_vix_trades.csv / signal_vix_improvements.csv")


if __name__ == "__main__":
    main()
