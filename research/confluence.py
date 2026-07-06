"""Task 3: daily confluence rules on QQQ + volume-filter split tests.

A. Confluence backtests (same engine, daily QQQ, next-open fills):
   - IBS<0.20 alone / 5-day-low-close alone / both (confluence)
   - confluence with looser IBS (0.25)
B. Volume splits (signal-quality analysis, not a backtest):
   For TT-A and 5DayLow-A entry signals, split by signal-day volume vs 20d avg
   and compare 2-session binary WR and avg move.
"""
import math
import os

import numpy as np
import pandas as pd

from engine import load_symbol, run_bt, compute_stats

OUT = os.path.dirname(os.path.abspath(__file__))
df = load_symbol("QQQ")
df["vol20"] = df["volume"].rolling(20).mean()
df["volx"] = df["volume"] / df["vol20"]


def nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


RUNS = [
    ("IBS20_alone", dict(
        entry_fn=lambda r: r.ibs < 0.20,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)),
    ("5DayLow_alone", dict(
        entry_fn=lambda r: nn(r.lc5) and r.close <= r.lc5,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)),
    ("CONF_ibs20_5dl", dict(
        entry_fn=lambda r: nn(r.lc5) and r.ibs < 0.20 and r.close <= r.lc5,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)),
    ("CONF_ibs25_5dl", dict(
        entry_fn=lambda r: nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)),
    ("CONF_ibs20_5dl_volhi", dict(
        entry_fn=lambda r: nn(r.lc5, r.volx) and r.ibs < 0.20 and r.close <= r.lc5 and r.volx > 1.2,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)),
]

print("=== A. Confluence backtests (QQQ daily, next-open fills) ===")
print(f"{'run':<22s} {'CAGR':>7s} {'maxDD':>7s} {'Sharpe':>6s} {'WR':>6s} {'PF':>5s} {'#tr':>5s} {'avgtr':>8s}")
for run_id, kw in RUNS:
    eq, trades = run_bt(df, **kw)
    st = compute_stats(eq, trades, run_id)
    print(f"{run_id:<22s} {st['cagr']:>7.1%} {st['maxdd']:>7.1%} {st['sharpe']:>6.2f} "
          f"{st['wr']:>6.1%} {st['pf']:>5.2f} {st['n_trades']:>5d} {st['avg_trade']:>8.3%}")

# ---- B. volume splits on signal quality ----
print("\n=== B. Volume splits: 2-session binary WR from next-open entry ===")
o = df["open"].values
c = df["close"].values
n = len(df)
sig_defs = {
    "TT_A (Mon down)": (df["weekday"] == 0) & (df["close"] < df["open"]),
    "5DayLow+IBS25": (df["ibs"] < 0.25) & (df["close"] <= df["lc5"]),
    "IBS<0.20": df["ibs"] < 0.20,
}
warm = df["date"] >= pd.Timestamp("2017-04-01")
print(f"{'signal':<18s} {'vol split':<12s} {'n':>5s} {'WR2d':>6s} {'avg2d':>8s}")
for name, sig in sig_defs.items():
    for split, mask in [("volx>1.2", df["volx"] > 1.2), ("volx<=1.2", df["volx"] <= 1.2)]:
        rows = df.index[(sig & mask & warm).fillna(False)]
        rets = []
        for i in rows:
            if i + 2 < n:
                entry = o[i + 1] * 1.0002
                rets.append(c[i + 2] / entry - 1)
        if not rets:
            continue
        a = np.array(rets)
        print(f"{name:<18s} {split:<12s} {len(a):>5d} {(a>0).mean():>6.1%} {a.mean():>8.3%}")
