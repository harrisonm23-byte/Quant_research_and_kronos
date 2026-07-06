"""Daily strategy suite on TQQQ and SQQQ, clean-table output."""
import math
import os

import numpy as np
import pandas as pd

from engine import load_symbol, run_bt

OUT = os.path.dirname(os.path.abspath(__file__))


def nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


STRATS = [
    ("DoubleSeven", dict(
        entry_fn=lambda r: nn(r.sma200, r.lc7) and r.close > r.sma200 and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7)),
    ("D7-NoFilter", dict(
        entry_fn=lambda r: nn(r.lc7) and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7)),
    ("IBS<.20/.70", dict(
        entry_fn=lambda r: r.ibs < 0.20,
        exit_fn=lambda r: r.ibs > 0.70)),
    ("5DayLow-A", dict(
        entry_fn=lambda r: nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)),
    ("TT-A (Mon)", dict(
        entry_fn=lambda r: r.weekday == 0 and r.close < r.open,
        exit_fn=None, max_hold=1)),
    ("TT-C (Mon)", dict(
        entry_fn=lambda r: (r.weekday == 0 and nn(r.prev_close, r.prev2_close)
                            and r.close < r.prev_close and r.prev_close < r.prev2_close),
        exit_fn=lambda r: nn(r.prev_high) and r.close > r.prev_high, max_hold=5)),
    ("LowerBand-A", dict(
        entry_fn=lambda r: nn(r.lower_band) and r.close < r.lower_band and r.ibs < 0.30,
        exit_fn=lambda r: nn(r.prev_high) and r.close > r.prev_high)),
]


def dd_stats(eq):
    peak = eq.cummax()
    dd = eq / peak - 1
    depths, cur, in_ep = [], 0.0, False
    for v in dd.values:
        if v < 0:
            in_ep, cur = True, min(cur, v)
        elif in_ep:
            depths.append(cur)
            cur, in_ep = 0.0, False
    if in_ep:
        depths.append(cur)
    depths = np.array(depths) if depths else np.array([0.0])
    return dd.min(), depths.mean(), np.median(depths)


rows = []
for sym in ["TQQQ", "SQQQ"]:
    df = load_symbol(sym)
    for name, kw in STRATS:
        eq, tr = run_bt(df, **kw)
        if not len(tr):
            rows.append([name, sym, None] + [np.nan] * 10)
            continue
        days = (eq.index[-1] - eq.index[0]).days
        ann = (eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1
        rets = tr["ret"].values
        wins, losses = rets[rets > 0], rets[rets <= 0]
        wr = len(wins) / len(rets)
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        maxdd, avgdd, meddd = dd_stats(eq)
        rows.append([name, sym, ann * 100, wr * 100, pf, maxdd * 100, avgdd * 100,
                     meddd * 100, rets.mean() * 100, np.median(rets) * 100,
                     tr["hold_days"].mean(), len(rets)])
    # buy & hold
    d = df[df["date"] >= "2017-04-01"]
    eq = pd.Series(d["close"].values, index=d["date"])
    days = (eq.index[-1] - eq.index[0]).days
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1
    maxdd, avgdd, meddd = dd_stats(eq)
    rows.append(["Buy&Hold", sym, ann * 100, np.nan, np.nan, maxdd * 100, avgdd * 100,
                 meddd * 100, np.nan, np.nan, np.nan, np.nan])

hdr = (f"{'Strategy':<12s}{'Sym':<6s}{'Ann%':>7s}{'WR%':>6s}{'PF':>6s}{'MaxDD':>7s}"
       f"{'AvgDD':>7s}{'MedDD':>7s}{'AvgTr':>7s}{'MedTr':>7s}{'Hold':>6s}{'#Tr':>5s}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    def f(x, w, dec=1):
        return f"{'--':>{w}s}" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:>{w}.{dec}f}"
    n_str = f"{'--':>5s}" if (r[11] is None or (isinstance(r[11], float) and np.isnan(r[11]))) else f"{int(r[11]):>5d}"
    print(f"{r[0]:<12s}{r[1]:<6s}" + f(r[2], 7) + f(r[3], 6) + f(r[4], 6, 2) + f(r[5], 7)
          + f(r[6], 7, 2) + f(r[7], 7, 2) + f(r[8], 7, 3) + f(r[9], 7, 3) + f(r[10], 6) + n_str)
print("-" * len(hdr))
print("Window 2017-04-01 -> 2026-07-01 | daily bars, fill next open, slippage 0.02%/side")
