#!/usr/bin/env python3
"""Check and paper-log operational QQQ/TQQQ intraday BB-fade candidates.

This runner intentionally uses the validated strategy semantics:
  * first eligible L1/L2/L3 signal per session;
  * entry at the next 5m open;
  * one risk cluster per symbol/setup (120m and EOD are virtual variants);
  * same-session 120-minute or EOD exit plans;
  * options overlays are paper plans, never automatic broker orders.

Usage:
  python3 intraday_strategy_runner.py list
  python3 intraday_strategy_runner.py check --symbols QQQ,TQQQ
  python3 intraday_strategy_runner.py log --symbols QQQ,TQQQ
  python3 intraday_strategy_runner.py status
  python3 intraday_strategy_runner.py close --id 1 --exit-ret 0.002
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, OUT)

import signal_exit_mechanics as exits
import signal_htf_combo as htf
from intraday_strategy_registry import BY_ID, OVERLAYS, STRATEGIES, rows

NY = ZoneInfo("America/New_York")
LOG = os.path.join(OUT, "intraday_strategy_paper.csv")
LOG_COLUMNS = [
    "id", "logged_at", "symbol", "setup", "strategy_id", "risk_cluster",
    "tier", "signal_ts", "signal_close", "entry_rule", "exit_mechanic",
    "planned_exit_ts", "status", "underlying_exit_ret", "overlay_ids",
    "overlay_status", "notes",
]


def parse_symbols(value):
    symbols = tuple(x.strip().upper() for x in value.split(",") if x.strip())
    bad = sorted(set(symbols) - {"QQQ", "TQQQ"})
    if bad:
        raise ValueError(f"unsupported symbols: {bad}")
    return symbols


def strategy_exit_ts(signal_ts, mechanic):
    """Planned wall-clock exit, capped at the final regular-session bar."""
    ts = pd.Timestamp(signal_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize(NY)
    else:
        ts = ts.tz_convert(NY)
    eod = ts.replace(hour=15, minute=55, second=0, microsecond=0)
    if mechanic == "fixed_eod":
        return eod
    if mechanic == "fixed_24":
        # Entry is next open and the 24th close is signal time + 120 minutes.
        return min(ts + pd.Timedelta(minutes=120), eod)
    raise ValueError(mechanic)


def latest_first_signals(symbol):
    panel = htf.build_panel(symbol)
    masks, _ = exits.setup_masks(panel)
    latest = panel.iloc[-1]
    hits = []
    for setup in ("L1", "L2", "L3"):
        if bool(masks[f"{setup}_first"].iloc[-1]):
            hits.append({
                "symbol": symbol,
                "setup": setup,
                "signal_ts": latest["ts"],
                "signal_close": float(latest["close"]),
                "rsi": float(latest["rsi"]),
                "vwap_dist": float(latest["vwap_dist"]),
            })
    return hits, latest["ts"]


def expand_strategies(hits, include_watch=False):
    expanded = []
    for hit in hits:
        for spec in STRATEGIES:
            if spec.symbol != hit["symbol"] or spec.setup != hit["setup"]:
                continue
            if spec.status == "watch" and not include_watch:
                continue
            expanded.append({
                **hit,
                "strategy_id": spec.strategy_id,
                "risk_cluster": spec.risk_cluster,
                "tier": spec.status,
                "exit_mechanic": spec.exit_mechanic,
                "planned_exit_ts": strategy_exit_ts(
                    hit["signal_ts"], spec.exit_mechanic
                ),
                "overlay_ids": ",".join(spec.overlay_ids),
            })
    return expanded


def load_log():
    if not os.path.exists(LOG):
        return pd.DataFrame(columns=LOG_COLUMNS)
    log = pd.read_csv(LOG)
    for col in LOG_COLUMNS:
        if col not in log:
            log[col] = ""
    return log[LOG_COLUMNS]


def print_registry():
    df = pd.DataFrame(rows())
    print(df[[
        "strategy_id", "status", "risk_cluster", "exit_mechanic",
        "overlay_ids", "evidence",
    ]].to_string(index=False))
    print("\nOverlay plans:")
    for overlay in OVERLAYS.values():
        width = f", width={overlay.width:.1%}" if overlay.width else ""
        print(
            f"  {overlay.overlay_id}: {overlay.structure}, {overlay.dte} DTE, "
            f"moneyness={overlay.moneyness:.1%}{width}, "
            f"cap=${overlay.premium_cap_usd}, {overlay.status}"
        )
    print("\nRisk rule: variants with the same risk_cluster are alternatives, not additive.")


def check(symbols, include_watch=False):
    all_hits = []
    now = pd.Timestamp.now(tz=NY)
    for symbol in symbols:
        hits, data_ts = latest_first_signals(symbol)
        age = now - pd.Timestamp(data_ts).tz_convert(NY)
        stale = age > pd.Timedelta(hours=24)
        print(
            f"{symbol}: latest bar {data_ts}"
            + (f"  [STALE {age.days}d]" if stale else "")
        )
        if not hits:
            print("  no first L1/L2/L3 signal on latest bar")
            continue
        expanded = expand_strategies(hits, include_watch=include_watch)
        if not expanded and hits:
            print("  signal exists, but strategies are WATCH; pass --include-watch")
        for row in expanded:
            print(
                f"  [{row['tier'].upper()}] {row['strategy_id']} "
                f"signal={row['signal_ts']} close={row['signal_close']:.2f} "
                f"planned_exit={row['planned_exit_ts']} "
                f"overlays={row['overlay_ids']}"
            )
        all_hits.extend(expanded)
    return all_hits


def log_hits(hits):
    if not hits:
        print("Nothing to log.")
        return
    log = load_log()
    existing = set(zip(
        log["strategy_id"].astype(str), log["signal_ts"].astype(str)
    ))
    next_id = int(pd.to_numeric(log["id"], errors="coerce").max()) + 1 if len(log) else 1
    new_rows = []
    for hit in hits:
        key = (hit["strategy_id"], str(hit["signal_ts"]))
        if key in existing:
            print(f"  skip duplicate {key[0]} @ {key[1]}")
            continue
        new_rows.append({
            "id": next_id,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "symbol": hit["symbol"],
            "setup": hit["setup"],
            "strategy_id": hit["strategy_id"],
            "risk_cluster": hit["risk_cluster"],
            "tier": hit["tier"],
            "signal_ts": str(hit["signal_ts"]),
            "signal_close": hit["signal_close"],
            "entry_rule": "next_5m_open",
            "exit_mechanic": hit["exit_mechanic"],
            "planned_exit_ts": str(hit["planned_exit_ts"]),
            "status": "open",
            "underlying_exit_ret": "",
            "overlay_ids": hit["overlay_ids"],
            "overlay_status": "planned",
            "notes": "virtual paper variant; do not stack same risk_cluster",
        })
        print(f"  LOG #{next_id} {hit['strategy_id']}")
        next_id += 1
    if new_rows:
        pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)[
            LOG_COLUMNS
        ].to_csv(LOG, index=False)
        print(f"Wrote {LOG} ({len(new_rows)} new)")


def close_row(row_id, exit_ret, notes):
    log = load_log()
    match = log.index[pd.to_numeric(log["id"], errors="coerce") == row_id]
    if not len(match):
        raise ValueError(f"id {row_id} not found")
    i = match[0]
    log.loc[i, "status"] = "closed"
    log.loc[i, "underlying_exit_ret"] = exit_ret
    if notes:
        log.loc[i, "notes"] = notes
    log.to_csv(LOG, index=False)
    print(f"Closed #{row_id}: underlying_exit_ret={exit_ret:+.3%}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["list", "check", "log", "status", "close"])
    ap.add_argument("--symbols", default="QQQ,TQQQ")
    ap.add_argument(
        "--include-watch", action="store_true",
        help="Include provisional TQQQ strategies",
    )
    ap.add_argument("--id", type=int)
    ap.add_argument("--exit-ret", type=float, default=0.0)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    if args.cmd == "list":
        print_registry()
    elif args.cmd in {"check", "log"}:
        hits = check(parse_symbols(args.symbols), include_watch=args.include_watch)
        if args.cmd == "log":
            log_hits(hits)
    elif args.cmd == "status":
        log = load_log()
        print("Paper log empty." if not len(log) else log.to_string(index=False))
    elif args.cmd == "close":
        if args.id is None:
            ap.error("close requires --id")
        close_row(args.id, args.exit_ret, args.notes)


if __name__ == "__main__":
    main()

