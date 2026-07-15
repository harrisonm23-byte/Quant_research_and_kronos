#!/usr/bin/env python3
"""Paper log for BB-fade keepers that clear a multi-metric gate (not WR alone).

Why not "anything over X% WR"?
  - WR without n overfits (10 trades at 90% is noise)
  - WR without avg/MFE ignores payoff (options need snap size)
  - WR without walk-forward ignores regime shift

Default gate (override via flags):
  n >= 15, WR >= 60%, avg >= +0.05%, both WF halves WR>=50% or avg>=0

Qualified by default under that gate: L1, L2, L3, L1v/L2v/L3v, L5
  Nesting: L1 ⊂ L2/L3/L1v — paper log suppresses parents when a child fires
  (so L1 does not steal L2/L3 trades). L2 and L3 are siblings (both may log).

Usage:
  python3 signal_keepers_paper.py gate          # show who qualifies + nest stats
  python3 signal_keepers_paper.py check         # scan latest bars, print armed
  python3 signal_keepers_paper.py log           # append armed keepers to paper CSV
  python3 signal_keepers_paper.py status        # show open/recent paper rows
  python3 signal_keepers_paper.py close --id ID --exit-ret 0.0015
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)
import signal_combo_scan as s
import signal_combo_phase3 as p3
import signal_keepers as sk

LOG = os.path.join(OUT, "signal_keepers_paper.csv")
HOLD = 5

# Default multi-metric gate — NOT win-rate alone
MIN_N = 15
MIN_WR = 0.60
MIN_AVG = 0.0005          # +0.05% underlying avg
REQUIRE_WF = True

# Optional looser "watch" tier (logged with tier=watch, not trade)
WATCH_MIN_N = 12
WATCH_MIN_WR = 0.55
WATCH_MIN_AVG = 0.0003


KEEPER_SPECS = [
    ("5m", "L1_5m_bbdn_prior_up", ["prior_up"], "bb_dn"),
    ("5m", "L2_5m_bbdn_prior_up_hvol", ["prior_up", "high_vol"], "bb_dn"),
    ("5m", "L3_5m_bbdn_prior_up_rsi35", ["prior_up", "rsi35"], "bb_dn"),
    ("5m", "L1v_5m_bbdn_prior_up_vix5up", ["prior_up", "vix5_rising"], "bb_dn"),
    ("5m", "L2v_5m_bbdn_prior_up_hvol_vix5up", ["prior_up", "high_vol", "vix5_rising"], "bb_dn"),
    ("5m", "L3v_5m_bbdn_prior_up_rsi35_vix5up", ["prior_up", "rsi35", "vix5_rising"], "bb_dn"),
    ("5m", "L1m_5m_bbdn_prior_up_vix_ma10", ["prior_up", "vix_above_ma10"], "bb_dn"),
    # HTF-confirmed (need build_panel flags on 5m)
    ("5m", "L1h_5m_prior_up_1h_below_sma9", ["prior_up", "1h_below_sma9"], "bb_dn"),
    ("5m", "L1c_5m_prior_up_15m_candle_dn", ["prior_up", "15m_candle_dn"], "bb_dn"),
    ("5m", "L1cw_5m_prior_up_15m_cdn_1w_sma9",
     ["prior_up", "15m_candle_dn", "1w_above_sma9"], "bb_dn"),
    ("5m", "L1w_5m_prior_up_1w_above_sma9", ["prior_up", "1w_above_sma9"], "bb_dn"),
    ("5m", "L2h_5m_hvol_15m_candle_dn", ["prior_up", "high_vol", "15m_candle_dn"], "bb_dn"),
    ("5m", "L3h_5m_rsi35_15m_candle_dn", ["prior_up", "rsi35", "15m_candle_dn"], "bb_dn"),
    ("5m", "L3a_5m_rsi35_1h_above_sma9", ["prior_up", "rsi35", "1h_above_sma9"], "bb_dn"),
    ("5m", "L3w_5m_rsi35_1w_above_sma9", ["prior_up", "rsi35", "1w_above_sma9"], "bb_dn"),
    ("5m", "L3cw_5m_rsi35_15m_below_1w_sma9",
     ["prior_up", "rsi35", "15m_below_sma9", "1w_above_sma9"], "bb_dn"),
    ("15m", "L4_15m_bbdn_stretch035", ["stretch035"], "bb_dn"),
    ("15m", "L5_15m_bbdn_rsi30", ["rsi30"], "bb_dn"),
    ("15m", "S1_15m_bbup_vwap_rsi65", ["stretch_ok_short", "rsi65"], "bb_up"),
    ("15m", "S1v_15m_bbup_vwap_rsi65_vix5crush", ["stretch_ok_short", "rsi65", "vix5_crush"], "bb_up"),
    ("15m", "S1d_15m_bbup_vwap_rsi65_vix_dn", ["stretch_ok_short", "rsi65", "vix_dn_day"], "bb_up"),
    ("15m", "S2_15m_bbup_gap_up_stretch025", ["gap_up", "stretch025"], "bb_up"),
    ("5m", "S3_5m_bbup_prior_down_rsi65", ["prior_down", "rsi65"], "bb_up"),
    ("15m", "S4_15m_bbup_narrow_bb", ["narrow_bb"], "bb_up"),
]

# Nested families: child suppresses parent on the same bar.
# L1 is the broad base; L2/L3/L1v/L1m are stricter; L2v/L3v are stricter still.
# L2 and L3 are siblings (not nested) — both can log if both fire.
NEST_CHILDREN = {
    "L1_5m_bbdn_prior_up": [
        "L2_5m_bbdn_prior_up_hvol",
        "L3_5m_bbdn_prior_up_rsi35",
        "L1v_5m_bbdn_prior_up_vix5up",
        "L1m_5m_bbdn_prior_up_vix_ma10",
        "L1h_5m_prior_up_1h_below_sma9",
        "L1c_5m_prior_up_15m_candle_dn",
        "L1w_5m_prior_up_1w_above_sma9",
        "L1cw_5m_prior_up_15m_cdn_1w_sma9",
    ],
    "L1c_5m_prior_up_15m_candle_dn": [
        "L1cw_5m_prior_up_15m_cdn_1w_sma9",
        "L2h_5m_hvol_15m_candle_dn",
        "L3h_5m_rsi35_15m_candle_dn",
    ],
    "L1w_5m_prior_up_1w_above_sma9": [
        "L1cw_5m_prior_up_15m_cdn_1w_sma9",
        "L3w_5m_rsi35_1w_above_sma9",
        "L3cw_5m_rsi35_15m_below_1w_sma9",
    ],
    "L2_5m_bbdn_prior_up_hvol": [
        "L2v_5m_bbdn_prior_up_hvol_vix5up",
        "L2h_5m_hvol_15m_candle_dn",
    ],
    "L3_5m_bbdn_prior_up_rsi35": [
        "L3v_5m_bbdn_prior_up_rsi35_vix5up",
        "L3h_5m_rsi35_15m_candle_dn",
        "L3a_5m_rsi35_1h_above_sma9",
        "L3w_5m_rsi35_1w_above_sma9",
        "L3cw_5m_rsi35_15m_below_1w_sma9",
    ],
    "L3h_5m_rsi35_15m_candle_dn": [
        "L3cw_5m_rsi35_15m_below_1w_sma9",
    ],
    "L3w_5m_rsi35_1w_above_sma9": [
        "L3cw_5m_rsi35_15m_below_1w_sma9",
    ],
    "L1v_5m_bbdn_prior_up_vix5up": [
        "L2v_5m_bbdn_prior_up_hvol_vix5up",
        "L3v_5m_bbdn_prior_up_rsi35_vix5up",
    ],
    "S1_15m_bbup_vwap_rsi65": [
        "S1v_15m_bbup_vwap_rsi65_vix5crush",
        "S1d_15m_bbup_vwap_rsi65_vix_dn",
    ],
}

# Priority within a nest (higher wins). Used when resolving same-bar conflicts.
PRIORITY = {
    "L3cw_5m_rsi35_15m_below_1w_sma9": 58,
    "L3v_5m_bbdn_prior_up_rsi35_vix5up": 50,
    "L2v_5m_bbdn_prior_up_hvol_vix5up": 45,
    "L3h_5m_rsi35_15m_candle_dn": 44,
    "L3a_5m_rsi35_1h_above_sma9": 43,
    "L3w_5m_rsi35_1w_above_sma9": 42,
    "L1cw_5m_prior_up_15m_cdn_1w_sma9": 41,
    "L3_5m_bbdn_prior_up_rsi35": 40,
    "L2h_5m_hvol_15m_candle_dn": 38,
    "L2_5m_bbdn_prior_up_hvol": 35,
    "L1c_5m_prior_up_15m_candle_dn": 33,
    "L1h_5m_prior_up_1h_below_sma9": 32,
    "L1v_5m_bbdn_prior_up_vix5up": 30,
    "L1w_5m_prior_up_1w_above_sma9": 28,
    "L1m_5m_bbdn_prior_up_vix_ma10": 25,
    "L1_5m_bbdn_prior_up": 10,
    "S1v_15m_bbup_vwap_rsi65_vix5crush": 40,
    "S1d_15m_bbup_vwap_rsi65_vix_dn": 35,
    "S1_15m_bbup_vwap_rsi65": 10,
}


def suppress_nested(hits):
    """Drop parent keepers when a child is also armed on the same ts/tf/side.

    L1 is suppressed by L2/L3/L1v/L1m; L2 by L2v; L3 by L3v; S1 by S1v/S1d.
    L2 and L3 are siblings — both may remain if both fire.
    """
    if not hits:
        return hits
    armed = {(h["tf"], h["side"], str(h["ts"]), h["keeper"]) for h in hits}
    keep = []
    suppressed = []
    for h in hits:
        children = NEST_CHILDREN.get(h["keeper"], [])
        if any((h["tf"], h["side"], str(h["ts"]), c) in armed for c in children):
            suppressed.append(h["keeper"])
            continue
        keep.append(h)

    # Within S1* / L1v* sibling VIX variants on same bar, keep highest PRIORITY only
    rival_groups = {
        "S1_rival": {"S1v_15m_bbup_vwap_rsi65_vix5crush", "S1d_15m_bbup_vwap_rsi65_vix_dn"},
        "Lv_rival": {"L2v_5m_bbdn_prior_up_hvol_vix5up", "L3v_5m_bbdn_prior_up_rsi35_vix5up"},
    }
    final = []
    drop = set()
    for h in keep:
        for gname, rivals in rival_groups.items():
            if h["keeper"] not in rivals:
                continue
            peers = [x for x in keep
                     if x["keeper"] in rivals
                     and x["tf"] == h["tf"] and x["side"] == h["side"]
                     and str(x["ts"]) == str(h["ts"])]
            best = max(peers, key=lambda x: PRIORITY.get(x["keeper"], 0))
            if h["keeper"] != best["keeper"]:
                drop.add(id(h))
                suppressed.append(h["keeper"])
    final = [h for h in keep if id(h) not in drop]
    if suppressed:
        print(f"  nest-suppress: {sorted(set(suppressed))} (child/higher-priority armed)")
    return final


def exclusive_stats(frames, min_n=8):
    """Show L1-only vs L2/L3 nested performance (answers 'L1 steals the trade')."""
    print("\n" + "=" * 88)
    print("NEST EXCLUSIVITY — parent-only bars (child did NOT also fire)")
    print("=" * 88)
    df = frames["5m"]
    specs = [
        ("L1", "L1_5m_bbdn_prior_up", [
            "L2_5m_bbdn_prior_up_hvol", "L3_5m_bbdn_prior_up_rsi35",
            "L1v_5m_bbdn_prior_up_vix5up", "L1m_5m_bbdn_prior_up_vix_ma10"]),
        ("L2", "L2_5m_bbdn_prior_up_hvol", ["L2v_5m_bbdn_prior_up_hvol_vix5up"]),
        ("L3", "L3_5m_bbdn_prior_up_rsi35", ["L3v_5m_bbdn_prior_up_rsi35_vix5up"]),
        ("L1v", "L1v_5m_bbdn_prior_up_vix5up", [
            "L2v_5m_bbdn_prior_up_hvol_vix5up", "L3v_5m_bbdn_prior_up_rsi35_vix5up"]),
    ]
    for label, name, children in specs:
        mask, side = sk.masks(df, name)
        child_any = pd.Series(False, index=df.index)
        for c in children:
            try:
                cm, _ = sk.masks(df, c)
                child_any = child_any | cm.fillna(False)
            except Exception:
                pass
        only = mask.fillna(False) & ~child_any
        _, r_all = s.backtest(df, mask, side, label=f"{label}|all", hold=HOLD)
        _, r_only = s.backtest(df, only, side, label=f"{label}|exclusive", hold=HOLD)
        wr = f"{r_all['wr']:.0%}" if r_all["n"] else "n/a"
        avg = f"{r_all['avg']:+.3%}" if r_all["n"] else "n/a"
        print(f"  {label:<4} all:        n={r_all['n']:>3} WR={wr} avg={avg}")
        if r_only["n"]:
            print(f"  {label:<4} exclusive: n={r_only['n']:>3} WR={r_only['wr']:.0%} "
                  f"avg={r_only['avg']:+.3%}  (no child fired)")
        else:
            print(f"  {label:<4} exclusive: n=0")
    m2, _ = sk.masks(df, "L2_5m_bbdn_prior_up_hvol")
    m3, _ = sk.masks(df, "L3_5m_bbdn_prior_up_rsi35")
    both = m2.fillna(False) & m3.fillna(False)
    _, rb = s.backtest(df, both, "long", label="L2∩L3", hold=HOLD)
    if rb["n"]:
        print(f"  L2∩L3 overlap: n={rb['n']:>3} WR={rb['wr']:.0%} avg={rb['avg']:+.3%}")
    else:
        print("  L2∩L3 overlap: n=0")
    print("  Paper rule: log most-specific child; suppress parent. L2∥L3 allowed.")


def load_frames():
    """5m panel with HTF states + VIX; 15m for L4/L5/S* keepers."""
    import signal_htf_combo as htf
    panel5 = htf.build_panel("SPY")
    # 15m frame for stretch/RSI keepers (enriched, no full HTF needed)
    raw5 = htf.load_5m("SPY")
    daily = htf.load_daily("SPY")
    frames = s.build_frames(raw5, daily)
    frames["5m"] = p3.enrich(panel5)
    for tf in ["15m", "30m", "1h"]:
        frames[tf] = p3.enrich(frames[tf])
    try:
        import signal_vix_study as vx
        vix_d = vx.prep_vix_daily(vx.fetch_vix_daily())
        try:
            vix_5m = vx.prep_vix_5m(vx.fetch_vix_5m())
        except Exception:
            vix_5m = None
        for tf in ["5m", "15m"]:
            frames[tf] = vx.align_vix(frames[tf], vix_d, vix_5m)
    except Exception as e:
        print(f"VIX attach skipped: {e}")
    return frames


def eval_keeper(frames, tf, name, parts, base, min_n, min_wr, min_avg, require_wf):
    df = frames[tf]
    mask, side = sk.masks(df, name)
    _, r = s.backtest(df, mask, side, label=name, hold=HOLD)
    wf = p3.walkforward_check(df, base, parts, side)
    h1, h2 = wf["H1"], wf["H2"]
    wf_ok = (
        h1["n"] >= 4 and h2["n"] >= 4
        and (h1["avg"] >= 0 or h1["wr"] >= 0.5)
        and (h2["avg"] >= 0 or h2["wr"] >= 0.5)
    )
    passes = (
        r["n"] >= min_n
        and not np.isnan(r["wr"]) and r["wr"] >= min_wr
        and not np.isnan(r["avg"]) and r["avg"] >= min_avg
        and (wf_ok if require_wf else True)
    )
    return dict(
        keeper=name, tf=tf, side=side, base=base,
        n=r["n"], wr=r["wr"], avg=r["avg"], med=r["med"],
        mfe_med=r["mfe_med"], hit15=r["hit15"],
        wf_ok=wf_ok,
        h1_n=h1["n"], h1_wr=h1["wr"], h1_avg=h1["avg"],
        h2_n=h2["n"], h2_wr=h2["wr"], h2_avg=h2["avg"],
        passes=passes,
    )


def classify(row, args):
    """trade / watch / skip."""
    if row["passes"]:
        return "trade"
    # watch tier
    watch = (
        row["n"] >= args.watch_min_n
        and not np.isnan(row["wr"]) and row["wr"] >= args.watch_min_wr
        and not np.isnan(row["avg"]) and row["avg"] >= args.watch_min_avg
        and row["wf_ok"]
    )
    if watch:
        return "watch"
    return "skip"


def cmd_gate(args):
    frames = load_frames()
    print("=" * 88)
    print(f"GATE: n>={args.min_n}  WR>={args.min_wr:.0%}  avg>={args.min_avg:.3%}  "
          f"WF={'on' if args.require_wf else 'off'}")
    print(f"WATCH: n>={args.watch_min_n}  WR>={args.watch_min_wr:.0%}  "
          f"avg>={args.watch_min_avg:.3%}  WF=on")
    print("=" * 88)
    rows = []
    for tf, name, parts, base in KEEPER_SPECS:
        side = "long" if base == "bb_dn" else "short"
        if side == "short" and not args.include_shorts and name.startswith("S"):
            # still evaluate but mark
            pass
        row = eval_keeper(frames, tf, name, parts, base,
                          args.min_n, args.min_wr, args.min_avg, args.require_wf)
        if side == "short" and not args.include_shorts:
            # shorts only eligible for watch unless --include-shorts
            tier = classify(row, args)
            if tier == "trade":
                tier = "watch"  # demote until explicitly included
            row["tier"] = tier
        else:
            row["tier"] = classify(row, args)
        rows.append(row)
        mark = {"trade": "TRADE", "watch": "watch", "skip": "skip "}[row["tier"]]
        wr = f"{row['wr']:.0%}" if not np.isnan(row["wr"]) else "n/a"
        avg = f"{row['avg']:+.3%}" if not np.isnan(row["avg"]) else "n/a"
        print(f"  [{mark}] {name:<36} n={row['n']:>3} WR={wr:>4} avg={avg:>8}  "
              f"WF={'Y' if row['wf_ok'] else 'N'}  "
              f"MFE med={row['mfe_med']:+.3%}" if not np.isnan(row["mfe_med"]) else
              f"  [{mark}] {name:<36} n={row['n']:>3}")

    out = pd.DataFrame(rows)
    path = os.path.join(OUT, "signal_keepers_gate.csv")
    out.to_csv(path, index=False)
    trade = out[out["tier"] == "trade"]["keeper"].tolist()
    watch = out[out["tier"] == "watch"]["keeper"].tolist()
    print(f"\nLog TRADE tier: {trade or '(none)'}")
    print(f"Log WATCH tier: {watch or '(none)'}")
    print(f"Wrote {path}")
    print("\nNote: L3 qualifies on WR+avg+WF — yes, log it. "
          "Do not auto-log every high-WR combo from the raw scan.")
    exclusive_stats(frames)
    return out


def armed_now(frames, keepers, dedup_nest=True):
    """Return armed keepers; by default suppress parents when children also fire."""
    hits = []
    for tf, name, parts, base in KEEPER_SPECS:
        if name not in keepers:
            continue
        df = frames[tf]
        mask, side = sk.masks(df, name)
        if not bool(mask.iloc[-1]):
            continue
        r = df.iloc[-1]
        hits.append(dict(
            keeper=name, side=side, tf=tf,
            ts=r["ts"], close=float(r["close"]),
            vwap_dist=float(r["vwap_dist"]) if pd.notna(r["vwap_dist"]) else np.nan,
            rsi=float(r["rsi"]) if pd.notna(r["rsi"]) else np.nan,
            prior_up=bool(r.get("prior_up", False)),
            high_vol=bool(r.get("high_vol", False)),
        ))
    if dedup_nest:
        hits = suppress_nested(hits)
    return hits


def load_log():
    if os.path.exists(LOG):
        return pd.read_csv(LOG)
    return pd.DataFrame(columns=[
        "id", "logged_at", "tier", "keeper", "side", "tf", "signal_ts",
        "close", "vwap_dist", "rsi", "status", "exit_ret", "notes",
    ])


def cmd_check(args):
    gate = cmd_gate(args)
    frames = load_frames()
    trade = set(gate.loc[gate["tier"] == "trade", "keeper"])
    watch = set(gate.loc[gate["tier"] == "watch", "keeper"])
    active = trade | (watch if args.log_watch else set())
    print("\n" + "=" * 88)
    print("ARMED NOW (qualified keepers, nest-deduped)" if not args.no_nest_dedup
          else "ARMED NOW (qualified keepers, raw — no nest dedup)")
    print("=" * 88)
    hits = armed_now(frames, active, dedup_nest=not args.no_nest_dedup)
    if not hits:
        print("  (none)")
    for h in hits:
        tier = "TRADE" if h["keeper"] in trade else "watch"
        print(f"  [{tier}] {h['keeper']} {h['side']} @ {h['ts']}  "
              f"c={h['close']:.2f} vwap={h['vwap_dist']:+.2%} RSI={h['rsi']:.0f}")
    return hits, gate


def cmd_log(args):
    hits, gate = cmd_check(args)
    if not hits:
        print("\nNothing to log.")
        return
    log = load_log()
    # dedupe: same keeper+signal_ts already open/logged
    existing = set()
    if len(log):
        existing = set(zip(log["keeper"].astype(str), log["signal_ts"].astype(str)))
    next_id = int(log["id"].max()) + 1 if len(log) and "id" in log.columns else 1
    trade = set(gate.loc[gate["tier"] == "trade", "keeper"])
    new_rows = []
    for h in hits:
        key = (h["keeper"], str(h["ts"]))
        if key in existing:
            print(f"  skip duplicate {h['keeper']} @ {h['ts']}")
            continue
        tier = "trade" if h["keeper"] in trade else "watch"
        if tier == "watch" and not args.log_watch:
            continue
        new_rows.append(dict(
            id=next_id, logged_at=datetime.now(timezone.utc).isoformat(),
            tier=tier, keeper=h["keeper"], side=h["side"], tf=h["tf"],
            signal_ts=str(h["ts"]), close=h["close"],
            vwap_dist=h["vwap_dist"], rsi=h["rsi"],
            status="open", exit_ret="", notes="",
        ))
        print(f"  LOG #{next_id} [{tier}] {h['keeper']} {h['side']} @ {h['ts']}")
        next_id += 1
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        log.to_csv(LOG, index=False)
        print(f"\nWrote {LOG} ({len(new_rows)} new)")
    else:
        print("\nNo new rows.")


def cmd_status(args):
    log = load_log()
    if not len(log):
        print("Paper log empty.")
        return
    print(log.to_string(index=False))
    open_n = (log["status"] == "open").sum()
    print(f"\nopen={open_n}  total={len(log)}")


def cmd_close(args):
    log = load_log()
    if not len(log):
        print("Paper log empty.")
        return
    idx = log.index[log["id"] == args.id]
    if not len(idx):
        print(f"id {args.id} not found")
        return
    i = idx[0]
    log.loc[i, "status"] = "closed"
    log.loc[i, "exit_ret"] = args.exit_ret
    if args.notes:
        log.loc[i, "notes"] = args.notes
    log.to_csv(LOG, index=False)
    print(f"Closed #{args.id} exit_ret={args.exit_ret}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gate", "check", "log", "status", "close"])
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--min-wr", type=float, default=MIN_WR)
    ap.add_argument("--min-avg", type=float, default=MIN_AVG)
    ap.add_argument("--watch-min-n", type=int, default=WATCH_MIN_N)
    ap.add_argument("--watch-min-wr", type=float, default=WATCH_MIN_WR)
    ap.add_argument("--watch-min-avg", type=float, default=WATCH_MIN_AVG)
    ap.add_argument("--require-wf", action=argparse.BooleanOptionalAction, default=REQUIRE_WF)
    ap.add_argument("--include-shorts", action="store_true",
                    help="Allow shorts into TRADE tier if they pass the gate")
    ap.add_argument("--log-watch", action="store_true",
                    help="Also log WATCH-tier armed signals")
    ap.add_argument("--no-nest-dedup", action="store_true",
                    help="Log all overlapping parents/children (not recommended)")
    ap.add_argument("--id", type=int, help="paper row id for close")
    ap.add_argument("--exit-ret", type=float, default=0.0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if args.cmd == "gate":
        cmd_gate(args)
    elif args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "log":
        cmd_log(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "close":
        if args.id is None:
            ap.error("close requires --id")
        cmd_close(args)


if __name__ == "__main__":
    main()
