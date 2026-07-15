#!/usr/bin/env python3
"""Paper log for BB-fade keepers that clear a multi-metric gate (not WR alone).

Why not "anything over X% WR"?
  - WR without n overfits (10 trades at 90% is noise)
  - WR without avg/MFE ignores payoff (options need snap size)
  - WR without walk-forward ignores regime shift

Default gate (override via flags):
  n >= 15, WR >= 60%, avg >= +0.05%, both WF halves WR>=50% or avg>=0

Qualified by default under that gate: L1, L2, L3, L4, L5
  (shorts usually fail avg/WR jointly — include with --include-shorts
   only if they still pass the gate)

Usage:
  python3 signal_keepers_paper.py gate          # show who qualifies
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
    ("15m", "L4_15m_bbdn_stretch035", ["stretch035"], "bb_dn"),
    ("15m", "L5_15m_bbdn_rsi30", ["rsi30"], "bb_dn"),
    ("15m", "S1_15m_bbup_vwap_rsi65", ["stretch_ok_short", "rsi65"], "bb_up"),
    ("15m", "S2_15m_bbup_gap_up_stretch025", ["gap_up", "stretch025"], "bb_up"),
    ("5m", "S3_5m_bbup_prior_down_rsi65", ["prior_down", "rsi65"], "bb_up"),
    ("15m", "S4_15m_bbup_narrow_bb", ["narrow_bb"], "bb_up"),
]


def load_frames():
    df5 = s.load_5m()
    daily = s.load_daily()
    frames = s.build_frames(df5, daily)
    for tf in ["5m", "15m", "30m", "1h"]:
        frames[tf] = p3.enrich(frames[tf])
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
    return out


def armed_now(frames, keepers):
    """Return list of (keeper, side, tf, ts, close, stretch, rsi) currently firing."""
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
    frames = load_frames()  # already loaded inside gate but reload cheap
    # reuse frames
    frames = load_frames()
    trade = set(gate.loc[gate["tier"] == "trade", "keeper"])
    watch = set(gate.loc[gate["tier"] == "watch", "keeper"])
    active = trade | (watch if args.log_watch else set())
    print("\n" + "=" * 88)
    print("ARMED NOW (qualified keepers only)")
    print("=" * 88)
    hits = armed_now(frames, active)
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
