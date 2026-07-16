"""OTM Scalp Signal — replicated spec + daily checker.

PRIMARY SIGNAL (replicated 2026-07-13):
  Name    : LB_sma300  ("Lower Band quiet bounce in long-term downtrend")
  Symbol  : QQQ
  Why     : Highest P(+0.75% within 2 sessions) = 83% (Wilson lo 72%)
             Best fit for cheap OTM call spike-exit overlay.

BACKUP SIGNAL (frequency + stability):
  Name    : TT_A  ("Turnaround Tuesday")
  Why     : n=178, P75=67% (lo 60%), stable both time halves, ~1.6 fires/month

Run:
  python3 otm_scalp_signal.py           # full replication report
  python3 otm_scalp_signal.py --check   # today's signal status
"""
import argparse
import math
import os
from datetime import date

import numpy as np
import pandas as pd

from engine import STAT_START, load_symbol

OUT = os.path.dirname(os.path.abspath(__file__))
SLIP = 0.0002

# ---- signal definitions ----
def _nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


def prep(sym):
    df = load_symbol(sym)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["volx"] = v / v.rolling(20).mean()
    hh10 = h.rolling(10).max()
    df["lower_band"] = hh10 - 2.5 * (h - l).rolling(25).mean()
    df["sma300"] = c.rolling(300).mean()
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    df["lc5"] = c.rolling(5).min()
    return df


def signal_lb_sma300(r, quiet=False):
    """Close below Connors lower band + IBS<0.30 + below SMA300."""
    if not _nn(r.lower_band, r.sma300):
        return False
    if not (r.close < r.lower_band and r.ibs < 0.30 and r.close < r.sma300):
        return False
    if quiet and not (r.volx <= 1.2):
        return False
    return True


def signal_tt_a(r):
    return r.weekday == 0 and r.close < r.open


def signal_5dl_quiet(r):
    return (_nn(r.lc5, r.volx) and r.ibs < 0.25
            and r.close <= r.lc5 and r.volx <= 1.2)


SIGNALS = {
    "LB_sma300": (signal_lb_sma300, dict(quiet=False)),
    "LB_quiet_sma300": (signal_lb_sma300, dict(quiet=True)),
    "TT_A": (signal_tt_a, {}),
    "5DayLow_quiet": (signal_5dl_quiet, {}),
}


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    mg = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr - mg) / den


def collect_trades(df, sig_fn, sig_kw=None):
    sig_kw = sig_kw or {}
    rows = list(df.itertuples(index=False))
    o, h, c = df["open"].values, df["high"].values, df["close"].values
    nb = len(df)
    start = pd.Timestamp(STAT_START)
    recs = []
    for i, r in enumerate(rows):
        if r.date < start or i >= nb - 3:
            continue
        if not sig_fn(r, **sig_kw):
            continue
        ep = o[i + 1] * (1 + SLIP)
        entry_date = df["date"].iloc[i + 1]
        window_h = h[i + 1:min(i + 3, nb)]
        hit75 = (window_h.max() / ep - 1) >= 0.0075 if len(window_h) else False
        hit100 = (window_h.max() / ep - 1) >= 0.010 if len(window_h) else False
        j2 = min(i + 2, nb - 1)
        win2 = c[j2] > ep
        recs.append(dict(signal_date=r.date, entry_date=entry_date, entry_px=ep,
                         hit75=hit75, hit100=hit100, win2=win2,
                         ret2=c[j2] / ep - 1))
    return pd.DataFrame(recs)


def half_stats(tr, col, lo, hi):
    sub = tr[(tr["entry_date"] >= lo) & (tr["entry_date"] < hi)]
    return len(sub), sub[col].mean() if len(sub) else float("nan")


# ---- OTM sim ----
def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, iv, kind="c"):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (0.04 + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    return S * _N(d1) - K * math.exp(-0.04 * T) * _N(d2)


def otm_call_pnl(df, tr, target=0.0075, otm=0.02, max_sess=2, expiry=5, contracts=2):
    rv = np.log(df["close"] / df["close"].shift(1)).rolling(20).std() * math.sqrt(252)
    idx = {d: i for i, d in enumerate(df["date"])}
    o, h, c = df["open"].values, df["high"].values, df["close"].values
    pnls = []
    for row in tr.itertuples():
        i = idx.get(row.entry_date)
        if i is None or math.isnan(rv.iloc[i]):
            continue
        S0 = row.entry_px / (1 + SLIP)  # unslipped ref for strike
        S0 = o[i]
        iv0 = rv.iloc[i]
        K = S0 * (1 + otm)
        spike_k = None
        for k in range(i, min(i + max_sess, len(h))):
            if h[k] >= S0 * (1 + target):
                spike_k = k
                break
        if spike_k is not None:
            Sx = S0 * (1 + target)
            Tx = max(expiry - (spike_k - i) - 0.5, 0.05) / 252.0
            ivx = iv0 * min(1.4, max(0.6, 1 - 3 * target))
        else:
            j = min(i + max_sess - 1, len(c) - 1)
            Sx = c[j]
            ret = Sx / S0 - 1
            Tx = max(expiry - max_sess, 0.05) / 252.0
            ivx = iv0 * min(1.4, max(0.6, 1 - 3 * ret))
        c0 = bs(S0, K, expiry / 252.0, iv0) * contracts
        c1 = bs(Sx, K, Tx, ivx) * contracts
        debit = c0 * 1.01
        if debit > 0:
            pnls.append((c1 * 0.99 - debit) / debit)
    return np.array(pnls)


def replicate(name, sym="QQQ"):
    fn, kw = SIGNALS[name]
    df = prep(sym)
    tr = collect_trades(df, fn, kw)
    if len(tr) < 10:
        print(f"{name}: insufficient trades ({len(tr)})")
        return
    n = len(tr)
    k75 = int(tr["hit75"].sum())
    k2 = int(tr["win2"].sum())
    years = (tr["entry_date"].max() - tr["entry_date"].min()).days / 365.25
    n1, h1_75 = half_stats(tr, "hit75", "2017-04-01", "2022-01-01")
    n2, h2_75 = half_stats(tr, "hit75", "2022-01-01", "2027-01-01")
    _, h1_2 = half_stats(tr, "win2", "2017-04-01", "2022-01-01")
    _, h2_2 = half_stats(tr, "win2", "2022-01-01", "2027-01-01")

    print(f"\n{'='*72}")
    print(f"REPLICATION: {name} on {sym}")
    print(f"{'='*72}")
    print(f"  Trades       : {n}  (~{n/years:.1f}/year)")
    print(f"  P(+0.75%/2d) : {tr['hit75'].mean():.1%}  Wilson-lo={wilson_lo(k75,n):.1%}")
    print(f"  P(+1.0%/2d)  : {tr['hit100'].mean():.1%}")
    print(f"  P(close>+@2d): {tr['win2'].mean():.1%}  Wilson-lo={wilson_lo(k2,n):.1%}")
    print(f"  Avg 2d ret   : {tr['ret2'].mean():+.3%}  median={tr['ret2'].median():+.3%}")
    print(f"  Time-split   : H1 n={n1} P75={h1_75:.0%} P2d={h1_2:.0%}  |  "
          f"H2 n={n2} P75={h2_75:.0%} P2d={h2_2:.0%}")

    for tgt, lbl in [(0.0075, "+0.75%"), (0.010, "+1.0%")]:
        pnls = otm_call_pnl(df, tr, target=tgt)
        if len(pnls):
            print(f"  OTM 2x 2%OTM call exit {lbl}: WR={(pnls>0).mean():.0%}  "
                  f"avg={pnls.mean():+.0%}  med={np.median(pnls):+.0%}")

    out = os.path.join(OUT, f"otm_scalp_{name}_{sym}.csv")
    tr.to_csv(out, index=False)
    print(f"  Log -> {out}")
    return tr


def check_today(sym="QQQ"):
    df = prep(sym)
    if len(df) < 2:
        print("No data")
        return
    r = df.iloc[-1]
    d = r["date"].date() if hasattr(r["date"], "date") else r["date"]
    print(f"\n=== OTM SCALP SIGNAL CHECK — {sym} {d} ===\n")
    fired = []
    for name, (fn, kw) in SIGNALS.items():
        if fn(r, **kw):
            fired.append(name)
    if not fired:
        print("  No scalp signals at today's close.")
    else:
        for name in fired:
            tag = " *** PRIMARY" if name.startswith("LB_") else " (backup)"
            print(f"  FIRE: {name}{tag}")
    print("""
If signal fired → enter next open:
  • Buy 2x QQQ calls ~2% OTM, ~5-day expiry
  • Exit when QQQ high >= entry +0.75% (limit/stop on option)
  • Hard time-stop: close of session 2
  • Skip new entries if QQQ close < SMA200 (optional gate)
""")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="Check today's signal only")
    p.add_argument("--sym", default="QQQ")
    args = p.parse_args()
    if args.check:
        check_today(args.sym)
        return

    print("OTM SCALP SIGNAL REPLICATION")
    print(f"Window: {STAT_START.date()} -> latest | 2-session scalp horizon\n")
    replicate("LB_sma300", args.sym)
    replicate("LB_quiet_sma300", args.sym)
    replicate("TT_A", args.sym)
    replicate("5DayLow_quiet", args.sym)

    print(f"\n{'='*72}")
    print("PROMOTED FOR OTM SCALPING: LB_sma300 (primary)")
    print(f"{'='*72}")
    print("""
ENTRY (all evaluated at daily close):
  • close < LowerBand  (HH10 - 2.5 * avg_range25)
  • IBS < 0.30
  • close < SMA300     (long-term downtrend regime — contrarian bounce)
  Optional filter: volume <= 1.2x 20d (LB_quiet_sma300) for higher P2d

OPTIONS EXPRESSION:
  • 2x 2% OTM calls, ~5 DTE
  • Exit on +0.75% underlying spike OR session-2 close
  • Do NOT hold to ETF median hold

BACKUP when LB_sma300 not firing: TT_A (Monday red → Tuesday open)
""")


if __name__ == "__main__":
    main()
