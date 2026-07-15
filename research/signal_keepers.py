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

    def col(c):
        return f[c].fillna(False) if c in f.columns else pd.Series(False, index=f.index)

    table = {
        "L1_5m_bbdn_prior_up": (first_dn & col("prior_up"), "long"),
        "L2_5m_bbdn_prior_up_hvol": (
            first_dn & col("prior_up") & col("high_vol"), "long"),
        "L3_5m_bbdn_prior_up_rsi35": (
            first_dn & col("prior_up") & col("rsi35"), "long"),
        "L4_15m_bbdn_stretch035": (
            first_dn & col("stretch035"), "long"),
        "L5_15m_bbdn_rsi30": (
            first_dn & col("rsi30"), "long"),
        # VIX-enhanced longs (from signal_vix_study)
        "L1v_5m_bbdn_prior_up_vix5up": (
            first_dn & col("prior_up") & col("vix5_rising"), "long"),
        "L2v_5m_bbdn_prior_up_hvol_vix5up": (
            first_dn & col("prior_up") & col("high_vol") & col("vix5_rising"), "long"),
        "L3v_5m_bbdn_prior_up_rsi35_vix5up": (
            first_dn & col("prior_up") & col("rsi35") & col("vix5_rising"), "long"),
        "L1m_5m_bbdn_prior_up_vix_ma10": (
            first_dn & col("prior_up") & col("vix_above_ma10"), "long"),
        # HTF-confirmed (from signal_htf_combo / consensus across SPY/QQQ/DIA/IWM)
        "L1h_5m_prior_up_1h_below_sma9": (
            first_dn & col("prior_up") & col("1h_below_sma9"), "long"),
        "L1c_5m_prior_up_15m_candle_dn": (
            first_dn & col("prior_up") & col("15m_candle_dn"), "long"),
        "L1cw_5m_prior_up_15m_cdn_1w_sma9": (
            first_dn & col("prior_up") & col("15m_candle_dn") & col("1w_above_sma9"), "long"),
        "L1w_5m_prior_up_1w_above_sma9": (
            first_dn & col("prior_up") & col("1w_above_sma9"), "long"),
        "L2h_5m_hvol_15m_candle_dn": (
            first_dn & col("prior_up") & col("high_vol") & col("15m_candle_dn"), "long"),
        "L3h_5m_rsi35_15m_candle_dn": (
            first_dn & col("prior_up") & col("rsi35") & col("15m_candle_dn"), "long"),
        "L3a_5m_rsi35_1h_above_sma9": (
            first_dn & col("prior_up") & col("rsi35") & col("1h_above_sma9"), "long"),
        "L3w_5m_rsi35_1w_above_sma9": (
            first_dn & col("prior_up") & col("rsi35") & col("1w_above_sma9"), "long"),
        "L3cw_5m_rsi35_15m_below_1w_sma9": (
            first_dn & col("prior_up") & col("rsi35") & col("15m_below_sma9")
            & col("1w_above_sma9"), "long"),
        "S1_15m_bbup_vwap_rsi65": (
            first_up & col("stretch_ok_short") & col("rsi65"), "short"),
        "S2_15m_bbup_gap_up_stretch025": (
            first_up & col("gap_up") & col("stretch025"), "short"),
        "S3_5m_bbup_prior_down_rsi65": (
            first_up & col("prior_down") & col("rsi65"), "short"),
        "S4_15m_bbup_narrow_bb": (
            first_up & col("narrow_bb"), "short"),
        # VIX-enhanced shorts
        "S1v_15m_bbup_vwap_rsi65_vix5crush": (
            first_up & col("stretch_ok_short") & col("rsi65") & col("vix5_crush"), "short"),
        "S1d_15m_bbup_vwap_rsi65_vix_dn": (
            first_up & col("stretch_ok_short") & col("rsi65") & col("vix_dn_day"), "short"),
    }
    return table[name]


PARTS = {
    "L1_5m_bbdn_prior_up": ["prior_up"],
    "L2_5m_bbdn_prior_up_hvol": ["prior_up", "high_vol"],
    "L3_5m_bbdn_prior_up_rsi35": ["prior_up", "rsi35"],
    "L4_15m_bbdn_stretch035": ["stretch035"],
    "L5_15m_bbdn_rsi30": ["rsi30"],
    "L1v_5m_bbdn_prior_up_vix5up": ["prior_up", "vix5_rising"],
    "L2v_5m_bbdn_prior_up_hvol_vix5up": ["prior_up", "high_vol", "vix5_rising"],
    "L3v_5m_bbdn_prior_up_rsi35_vix5up": ["prior_up", "rsi35", "vix5_rising"],
    "L1m_5m_bbdn_prior_up_vix_ma10": ["prior_up", "vix_above_ma10"],
    "L1h_5m_prior_up_1h_below_sma9": ["prior_up", "1h_below_sma9"],
    "L1c_5m_prior_up_15m_candle_dn": ["prior_up", "15m_candle_dn"],
    "L1cw_5m_prior_up_15m_cdn_1w_sma9": ["prior_up", "15m_candle_dn", "1w_above_sma9"],
    "L1w_5m_prior_up_1w_above_sma9": ["prior_up", "1w_above_sma9"],
    "L2h_5m_hvol_15m_candle_dn": ["prior_up", "high_vol", "15m_candle_dn"],
    "L3h_5m_rsi35_15m_candle_dn": ["prior_up", "rsi35", "15m_candle_dn"],
    "L3a_5m_rsi35_1h_above_sma9": ["prior_up", "rsi35", "1h_above_sma9"],
    "L3w_5m_rsi35_1w_above_sma9": ["prior_up", "rsi35", "1w_above_sma9"],
    "L3cw_5m_rsi35_15m_below_1w_sma9": ["prior_up", "rsi35", "15m_below_sma9", "1w_above_sma9"],
    "S1_15m_bbup_vwap_rsi65": ["stretch_ok_short", "rsi65"],
    "S2_15m_bbup_gap_up_stretch025": ["gap_up", "stretch025"],
    "S3_5m_bbup_prior_down_rsi65": ["prior_down", "rsi65"],
    "S4_15m_bbup_narrow_bb": ["narrow_bb"],
    "S1v_15m_bbup_vwap_rsi65_vix5crush": ["stretch_ok_short", "rsi65", "vix5_crush"],
    "S1d_15m_bbup_vwap_rsi65_vix_dn": ["stretch_ok_short", "rsi65", "vix_dn_day"],
}


def run_all(frames):
    specs = [
        ("5m", "L1_5m_bbdn_prior_up"),
        ("5m", "L2_5m_bbdn_prior_up_hvol"),
        ("5m", "L3_5m_bbdn_prior_up_rsi35"),
        ("5m", "L1v_5m_bbdn_prior_up_vix5up"),
        ("5m", "L2v_5m_bbdn_prior_up_hvol_vix5up"),
        ("5m", "L3v_5m_bbdn_prior_up_rsi35_vix5up"),
        ("5m", "L1m_5m_bbdn_prior_up_vix_ma10"),
        ("5m", "L1h_5m_prior_up_1h_below_sma9"),
        ("5m", "L1c_5m_prior_up_15m_candle_dn"),
        ("5m", "L1cw_5m_prior_up_15m_cdn_1w_sma9"),
        ("5m", "L1w_5m_prior_up_1w_above_sma9"),
        ("5m", "L2h_5m_hvol_15m_candle_dn"),
        ("5m", "L3h_5m_rsi35_15m_candle_dn"),
        ("5m", "L3a_5m_rsi35_1h_above_sma9"),
        ("5m", "L3w_5m_rsi35_1w_above_sma9"),
        ("5m", "L3cw_5m_rsi35_15m_below_1w_sma9"),
        ("15m", "L4_15m_bbdn_stretch035"),
        ("15m", "L5_15m_bbdn_rsi30"),
        ("15m", "S1_15m_bbup_vwap_rsi65"),
        ("15m", "S1v_15m_bbup_vwap_rsi65_vix5crush"),
        ("15m", "S1d_15m_bbup_vwap_rsi65_vix_dn"),
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
        parts = PARTS[name]
        # walk-forward only on non-VIX columns that exist in enrich(); VIX cols need presence
        wf_parts = [p for p in parts if p in df.columns]
        base = "bb_dn" if side == "long" else "bb_up"
        if wf_parts:
            wf = p3.walkforward_check(df, base, wf_parts, side)
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
            "L1v_5m_bbdn_prior_up_vix5up", "L2v_5m_bbdn_prior_up_hvol_vix5up",
            "L3v_5m_bbdn_prior_up_rsi35_vix5up", "L1m_5m_bbdn_prior_up_vix_ma10",
            "S3_5m_bbup_prior_down_rsi65",
        ]
        names_15m = [
            "L4_15m_bbdn_stretch035", "L5_15m_bbdn_rsi30",
            "S1_15m_bbup_vwap_rsi65", "S1v_15m_bbup_vwap_rsi65_vix5crush",
            "S1d_15m_bbup_vwap_rsi65_vix_dn",
            "S2_15m_bbup_gap_up_stretch025", "S4_15m_bbup_narrow_bb",
        ]
        for name in (names_5m if tf == "5m" else names_15m):
            mask, side = masks(df, name)
            if bool(mask.iloc[-1]):
                print(f"    *** {name} ({side})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", action="store_true")
    args = ap.parse_args()
    import signal_htf_combo as htf
    import signal_vix_study as vx
    panel5 = htf.build_panel("SPY")
    try:
        vix_d = vx.prep_vix_daily(vx.fetch_vix_daily())
        try:
            vix_5m = vx.prep_vix_5m(vx.fetch_vix_5m())
        except Exception:
            vix_5m = None
        panel5 = vx.align_vix(panel5, vix_d, vix_5m)
    except Exception as e:
        print(f"VIX attach skipped: {e}")
        vix_d, vix_5m = None, None
    raw5 = htf.load_5m("SPY")
    frames = {
        "5m": panel5,
        "15m": p3.enrich(s.prep(s.resample(raw5, "15min"), intraday=True)),
    }
    if vix_d is not None:
        try:
            frames["15m"] = vx.align_vix(frames["15m"], vix_d, vix_5m)
        except Exception:
            pass
    run_all(frames)
    if args.scan:
        scan(frames)


if __name__ == "__main__":
    main()
