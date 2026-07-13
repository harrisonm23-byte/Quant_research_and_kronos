#!/usr/bin/env python3
"""Starter OTM overlay — one signal (LB_quiet_sma300 on QQQ), paper log only.

Signal (at daily close):
  close < lower_band AND IBS < 0.30 AND close < SMA300 AND vol <= 1.2x20d

Options (next session):
  Buy 2× QQQ ATM (or lightly OTM) calls, ~1 DTE
  Take-profit: QQQ high >= entry +1.0%
  Stop: EOD only — if close <= entry -1.0% (do NOT use intraday stops;
        mean-reversion dips before bouncing)
  Time stop: close of session 2
  Hard risk cap: $250 premium = max loss

Note on OTM: price need NOT reach the strike — a move TOWARD the strike
still lifts OTM premium via delta/gamma. Far OTM (2%+) underperforms here
because typical MFE is ~2%, not 3%+. Prefer ATM to ~1% OTM.

Usage:
  python3 otm_overlay_starter.py check
  python3 otm_overlay_starter.py sim
  python3 otm_overlay_starter.py open --entry PRICE
  python3 otm_overlay_starter.py status
  python3 otm_overlay_starter.py close
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(OUT)
LOG = os.path.join(OUT, "otm_overlay_paper.csv")
SYMBOL = "QQQ"
OTM_PCT = 0.00          # ATM — must clear strike for real leverage
SPIKE_PCT = 0.010       # take-profit when underlying +1.0%
EOD_STOP_PCT = 0.010    # exit at close if underlying <= entry -1% (no intraday stop)
MAX_SESSIONS = 2
CONTRACTS = 2
EXPIRY_DTE = 1          # short-dated — gamma is the point
MAX_PREMIUM_USD = 250   # hard risk cap = true max loss
REQUIRE_QUIET = True
STAT_START = pd.Timestamp("2017-04-01")
R = 0.04
COST = 0.02


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
    ok = (_nn(r.lower_band, r.sma300)
          and r.close < r.lower_band and r.ibs < 0.30 and r.close < r.sma300)
    if ok and REQUIRE_QUIET:
        ok = _nn(r.volx) and r.volx <= 1.2
    return ok


def cmd_check():
    df = load()
    r = df.iloc[-1]
    d = r["date"].strftime("%Y-%m-%d")
    print(f"OTM overlay starter — {SYMBOL} bar {d}")
    print(f"  close={r['close']:.2f}  lower_band={r['lower_band']:.2f}  "
          f"IBS={r['ibs']:.2f}  SMA300={r['sma300']:.2f}  volx={r['volx']:.2f}")
    if signal_fired(r):
        est_strike = round(r["close"] * (1 + OTM_PCT), 0)
        print("\n  >>> SIGNAL ON — paper entry next open <<<")
        print(f"  Plan: {CONTRACTS}x ~{est_strike:.0f} call (~{OTM_PCT*100:.0f}% OTM), weekly expiry")
        print(f"  Exit: QQQ high >= entry×{1+SPIKE_PCT:.4f}  OR  session-{MAX_SESSIONS} close")
        print(f"  Size cap: ${MAX_PREMIUM_USD} total premium")
        print("\n  After open fills:  python3 otm_overlay_starter.py open --entry PRICE")
    else:
        print("\n  No signal. Flat.")


def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs(S, K, T, iv):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (R + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-R * T) * _N(d2)


def cmd_sim():
    """Historical BS sim of the starter playbook (quick spike exit)."""
    df = load()
    df["rv20"] = np.log(df["close"] / df["close"].shift(1)).rolling(20).std() * math.sqrt(252)
    o, h, c = df["open"].values, df["high"].values, df["close"].values
    rv = df["rv20"].values
    rows = list(df.itertuples(index=False))
    nb = len(df)

    pnls, holds, hits, years = [], [], [], []
    for i, r in enumerate(rows):
        if r.date < STAT_START or i >= nb - 3:
            continue
        if not signal_fired(r):
            continue
        j = i + 1
        if math.isnan(rv[j]) or rv[j] <= 0:
            continue
        S0, iv0 = o[j], rv[j]
        K = S0 * (1 + OTM_PCT)
        spike = None
        for k in range(j, min(j + MAX_SESSIONS, nb)):
            if h[k] >= S0 * (1 + SPIKE_PCT):
                spike = k
                break
        if spike is not None:
            Sx, ret, hit, held = S0 * (1 + SPIKE_PCT), SPIKE_PCT, True, spike - j + 1
            elapsed = (spike - j) + 0.5
        else:
            k = min(j + MAX_SESSIONS - 1, nb - 1)
            Sx, ret, hit, held = c[k], c[k] / S0 - 1, False, k - j + 1
            elapsed = MAX_SESSIONS
        T0 = EXPIRY_DTE / 252.0
        Tx = max(EXPIRY_DTE - elapsed, 0.05) / 252.0
        if EXPIRY_DTE <= 1 and spike is not None and elapsed >= EXPIRY_DTE:
            Tx = 0.02 / 252.0
        c0 = _bs(S0, K, T0, iv0) * (1 + COST / 2)
        ivx = iv0 * min(1.4, max(0.6, 1 - 3 * ret))
        c1 = _bs(Sx, K, Tx, ivx) * (1 - COST / 2)
        if c0 < 0.05:
            continue
        pnls.append((c1 - c0) / c0)
        holds.append(held)
        hits.append(hit)
        years.append(pd.Timestamp(df["date"].iloc[j]).year)

    a = np.array(pnls)
    strike_lbl = "ATM" if OTM_PCT == 0 else f"{OTM_PCT*100:.1f}% OTM"
    print(f"=== SIM: LB_quiet_sma300 | {strike_lbl} | {EXPIRY_DTE} DTE | "
          f"exit +{SPIKE_PCT*100:.2f}% / {MAX_SESSIONS}d ===")
    print(f"  n={len(a)}  WR={(a>0).mean():.0%}  avg={a.mean():+.1%}  med={np.median(a):+.1%}")
    print(f"  spike hit={np.mean(hits):.0%}  avgHold_hit="
          f"{np.mean([hh for hh, x in zip(holds, hits) if x]):.2f}d  "
          f"avgHold_miss={np.mean([hh for hh, x in zip(holds, hits) if not x]):.2f}d")
    print(f"  win avg={a[a>0].mean():+.1%}  loss avg={a[a<=0].mean():+.1%}")
    print(f"  leverage vs +{SPIKE_PCT*100:.2f}% spot ≈ {a.mean()/SPIKE_PCT:.1f}x")
    dollar = MAX_PREMIUM_USD * a
    print(f"  ${MAX_PREMIUM_USD}/trade: total P&L=${dollar.sum():+.0f}  "
          f"avg=${dollar.mean():+.1f}/trade")
    print(f"\n  {'year':>6s}{'n':>4s}{'WR':>6s}{'avg':>8s}{'$pnl':>8s}")
    for yr in sorted(set(years)):
        mask = np.array(years) == yr
        y = a[mask]
        print(f"  {yr:>6d}{len(y):>4d}{(y>0).mean():>6.0%}{y.mean():>+8.1%}{MAX_PREMIUM_USD*y.sum():>+8.0f}")
    print("\n  Note: BS + RV20 IV — not real chains. Exit target must CLEAR the strike.")


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
    p = argparse.ArgumentParser(description="Starter OTM overlay (LB_quiet_sma300 / QQQ)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="Check latest bar for signal")
    sub.add_parser("sim", help="Historical return simulation")
    o = sub.add_parser("open", help="Log paper entry")
    o.add_argument("--entry", type=float, default=0.0, help="Fill price (default: next open)")
    sub.add_parser("status", help="Open trades + spike level")
    sub.add_parser("close", help="Close open trade using exit rules")
    args = p.parse_args()
    {"check": cmd_check, "sim": cmd_sim, "open": lambda: cmd_open(args.entry),
     "status": cmd_status, "close": cmd_close}[args.cmd]()


if __name__ == "__main__":
    main()
