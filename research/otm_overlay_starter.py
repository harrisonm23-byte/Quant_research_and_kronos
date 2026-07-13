#!/usr/bin/env python3
"""Starter OTM overlay — one signal (LB_sma300 on QQQ), paper log only.

Signal (at daily close):
  close < lower_band AND IBS < 0.30 AND close < SMA300

Options (next session):
  Buy 2× QQQ calls ~2% OTM, nearest weekly (~5 DTE)
  Exit when QQQ high >= entry +0.75%, OR close of trade day 2

Usage:
  python3 otm_overlay_starter.py check          # signal tonight?
  python3 otm_overlay_starter.py open           # log paper entry (after signal)
  python3 otm_overlay_starter.py status         # open trades + exit levels
  python3 otm_overlay_starter.py close          # apply exits to paper log
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(OUT)
LOG = os.path.join(OUT, "otm_overlay_paper.csv")
SYMBOL = "QQQ"
OTM_PCT = 0.02
SPIKE_PCT = 0.0075
MAX_SESSIONS = 2
CONTRACTS = 2
MAX_PREMIUM_USD = 250  # starter size cap


def _nn(*v):
    return all(not (isinstance(x, float) and np.isnan(x)) for x in v)


def ensure_daily():
    path = os.path.join(OUT, f"{SYMBOL}_daily.csv")
    if os.path.exists(path):
        return path
    tmp = os.path.join(OUT, "_fetch_starter")
    os.makedirs(tmp, exist_ok=True)
    subprocess.check_call([
        sys.executable, os.path.join(REPO, "examples", "fetch_market_data.py"),
        SYMBOL, "--range", "10Y", "--outdir", tmp,
    ])
    df = pd.read_csv(os.path.join(tmp, f"{SYMBOL}.csv")).rename(columns={"timestamps": "date"})
    df[["date", "open", "high", "low", "close", "volume"]].to_csv(path, index=False)
    return path


def load():
    ensure_daily()
    df = pd.read_csv(os.path.join(OUT, f"{SYMBOL}_daily.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["volx"] = v / v.rolling(20).mean()
    df["lower_band"] = h.rolling(10).max() - 2.5 * (h - l).rolling(25).mean()
    df["sma300"] = c.rolling(300).mean()
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    return df


def signal_fired(r):
    return (_nn(r.lower_band, r.sma300)
            and r.close < r.lower_band and r.ibs < 0.30 and r.close < r.sma300)


def cmd_check():
    df = load()
    r = df.iloc[-1]
    d = r["date"].strftime("%Y-%m-%d")
    print(f"OTM overlay starter — {SYMBOL} bar {d}")
    print(f"  close={r['close']:.2f}  lower_band={r['lower_band']:.2f}  "
          f"IBS={r['ibs']:.2f}  SMA300={r['sma300']:.2f}")
    if signal_fired(r):
        est_strike = round(r["close"] * (1 + OTM_PCT), 0)
        print("\n  >>> SIGNAL ON — paper entry next open <<<")
        print(f"  Plan: {CONTRACTS}x ~{est_strike:.0f} call (~2% OTM), weekly expiry")
        print(f"  Exit: QQQ high >= entry×{1+SPIKE_PCT:.4f}  OR  session-{MAX_SESSIONS} close")
        print(f"  Size cap: ${MAX_PREMIUM_USD} total premium")
        print("\n  After open fills:  python3 otm_overlay_starter.py open --entry PRICE")
    else:
        print("\n  No signal. Flat.")


def _read_log():
    if not os.path.exists(LOG):
        return pd.DataFrame(columns=[
            "id", "signal_date", "entry_date", "entry_px", "strike", "spike_px",
            "status", "exit_date", "exit_px", "exit_reason", "sessions_held",
        ])
    return pd.read_csv(LOG, parse_dates=["signal_date", "entry_date", "exit_date"])


def _write_log(df):
    df.to_csv(LOG, index=False)


def cmd_open(entry_px: float):
    df = load()
    r = df.iloc[-2]  # signal bar = yesterday if logging after today's open
    # use latest bar where signal fired
    sig_i = None
    for i in range(len(df) - 1, max(len(df) - 10, -1), -1):
        if signal_fired(df.iloc[i]):
            sig_i = i
            break
    if sig_i is None:
        print("No recent signal in last 10 bars. Run `check` first.")
        sys.exit(1)
    sig = df.iloc[sig_i]
    entry_row = df.iloc[sig_i + 1] if sig_i + 1 < len(df) else None
    entry_date = entry_row["date"] if entry_row is not None else pd.NaT
    if entry_px <= 0:
        entry_px = float(entry_row["open"]) if entry_row is not None else float(sig["close"])

    log = _read_log()
    if len(log) and log["status"].eq("open").any():
        print("Already have an open paper trade. Close it first (`close`).")
        sys.exit(1)

    strike = round(entry_px * (1 + OTM_PCT), 0)
    spike_px = round(entry_px * (1 + SPIKE_PCT), 4)
    new_id = 1 if not len(log) else int(log["id"].max()) + 1
    row = dict(
        id=new_id, signal_date=sig["date"], entry_date=entry_date,
        entry_px=entry_px, strike=strike, spike_px=spike_px,
        status="open", exit_date=pd.NaT, exit_px=np.nan,
        exit_reason="", sessions_held=0,
    )
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    _write_log(log)
    print(f"Paper trade #{new_id} opened")
    print(f"  entry={entry_px:.2f}  strike≈{strike:.0f}  spike_target={spike_px:.2f}")
    print(f"  log: {LOG}")


def cmd_status():
    log = _read_log()
    if not len(log):
        print("No paper trades yet.")
        return
    df = load()
    last = df.iloc[-1]
    print(f"Latest {SYMBOL} bar: {last['date'].strftime('%Y-%m-%d')}  "
          f"high={last['high']:.2f}  close={last['close']:.2f}\n")
    for t in log.itertuples():
        flag = "OPEN" if t.status == "open" else "closed"
        print(f"#{t.id} [{flag}] signal={pd.Timestamp(t.signal_date).date()}  "
              f"entry={t.entry_px:.2f}  spike={t.spike_px:.2f}")
        if t.status == "open":
            hit = last["high"] >= t.spike_px
            print(f"     spike hit on last bar? {'YES' if hit else 'no'}  "
                  f"(need high >= {t.spike_px:.2f})")


def cmd_close():
    log = _read_log()
    open_tr = log[log["status"] == "open"]
    if not open_tr.empty and len(open_tr) > 1:
        print("Multiple open trades — close manually in CSV.")
        sys.exit(1)
    if open_tr.empty:
        print("No open paper trade.")
        return

    df = load()
    idx = {d: i for i, d in enumerate(df["date"])}
    t = open_tr.iloc[0]
    i = idx.get(pd.Timestamp(t["entry_date"]))
    if i is None:
        print("Entry date not in price data.")
        return

    o, h, c, dates = df["open"].values, df["high"].values, df["close"].values, df["date"].values
    entry_px = t["entry_px"]
    spike_px = t["spike_px"]
    exit_px, exit_date, reason, held = None, None, "", 0

    for sess in range(MAX_SESSIONS):
        k = i + sess
        if k >= len(df):
            break
        held = sess + 1
        if h[k] >= spike_px:
            exit_px, exit_date, reason = spike_px, dates[k], "spike"
            break
    if exit_px is None:
        k = min(i + MAX_SESSIONS - 1, len(df) - 1)
        exit_px, exit_date, reason = c[k], dates[k], "time"
        held = k - i + 1

    ret = exit_px / entry_px - 1
    log.loc[log["id"] == t["id"], ["status", "exit_date", "exit_px", "exit_reason", "sessions_held"]] = [
        "closed", exit_date, exit_px, reason, held,
    ]
    _write_log(log)
    print(f"Closed paper #{int(t['id'])}  reason={reason}  sessions={held}")
    print(f"  underlying ret={ret:+.2%}  (option P&L not modeled — log underlying only)")


def main():
    p = argparse.ArgumentParser(description="Starter OTM overlay (LB_sma300 / QQQ)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Check latest bar for signal")
    o = sub.add_parser("open", help="Log paper entry")
    o.add_argument("--entry", type=float, default=0.0, help="Fill price (default: next open)")
    sub.add_parser("status", help="Open trades + spike level")
    sub.add_parser("close", help="Close open trade using exit rules")
    args = p.parse_args()
    {"check": cmd_check, "open": lambda: cmd_open(args.entry),
     "status": cmd_status, "close": cmd_close}[args.cmd]()


if __name__ == "__main__":
    main()
