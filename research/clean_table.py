"""Consolidated Excel-style performance table for the daily strategy suite."""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    ("DoubleSeven",  "SPY", "S1_DoubleSeven_SPY"),
    ("DoubleSeven",  "QQQ", "S1_DoubleSeven_QQQ"),
    ("IBS<.20/.70",  "SPY", "S4_IBS_SPY_e20_x70"),
    ("IBS<.20/.70",  "QQQ", "S4_IBS_QQQ_e20_x70"),
    ("5DayLow-A",    "SPY", "S9_5DayLow_A_SPY"),
    ("5DayLow-A",    "QQQ", "S9_5DayLow_A_QQQ"),
    ("TripleRSI",    "SPY", "S7_TripleRSI_SPY"),
    ("TT-A (Mon)",   "SPY", "S6_TT_A_SPY"),
    ("TT-C (Mon)",   "SPY", "S6_TT_C_SPY"),
    ("LowerBand-B",  "QQQ", "S5_LowerBand_QQQ_B_sma300"),
    ("IBS+RSI21",    "SPY", "S8_IBSRSI21_SPY"),
    ("RSI2-Mod-A",   "SPY", "S3_RSI2Mod_SPY_A_nostop"),
]


def dd_stats(eq):
    peak = eq.cummax()
    dd = eq / peak - 1
    maxdd = dd.min()
    # episodes: contiguous dd<0 stretches -> depth of each
    depths = []
    cur = 0.0
    in_ep = False
    for v in dd.values:
        if v < 0:
            in_ep = True
            cur = min(cur, v)
        elif in_ep:
            depths.append(cur)
            cur = 0.0
            in_ep = False
    if in_ep:
        depths.append(cur)
    depths = np.array(depths) if depths else np.array([0.0])
    return maxdd, depths.mean(), np.median(depths)


rows = []
for name, sym, rid in RUNS:
    eq = pd.read_csv(os.path.join(OUT, f"equity_{rid}.csv"), index_col=0, parse_dates=True).iloc[:, 0]
    tr = pd.read_csv(os.path.join(OUT, f"trades_{rid}.csv"))
    days = (eq.index[-1] - eq.index[0]).days
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1
    rets = tr["ret"].values
    wins, losses = rets[rets > 0], rets[rets <= 0]
    wr = len(wins) / len(rets)
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
    maxdd, avgdd, meddd = dd_stats(eq)
    rows.append([name, sym, "1D", ann * 100, wr * 100, pf, maxdd * 100, avgdd * 100,
                 meddd * 100, rets.mean() * 100, np.median(rets) * 100,
                 tr["hold_days"].mean(), len(rets)])

# buy & hold benchmarks
for sym in ["SPY", "QQQ"]:
    d = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"), parse_dates=["date"])
    d = d[d["date"] >= "2017-04-01"]
    eq = pd.Series(d["close"].values, index=d["date"])
    days = (eq.index[-1] - eq.index[0]).days
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1
    maxdd, avgdd, meddd = dd_stats(eq)
    rows.append([f"Buy&Hold", sym, "1D", ann * 100, np.nan, np.nan, maxdd * 100,
                 avgdd * 100, meddd * 100, np.nan, np.nan, np.nan, np.nan])

hdr = (f"{'Strategy':<12s}{'Sym':<5s}{'TF':<4s}{'Ann%':>6s}{'WR%':>6s}{'PF':>6s}"
       f"{'MaxDD':>7s}{'AvgDD':>7s}{'MedDD':>7s}{'AvgTr':>7s}{'MedTr':>7s}{'Hold':>6s}{'#Tr':>5s}")
print(hdr)
print("-" * len(hdr))
for r in rows:
    def f(x, w, dec=1):
        return f"{'--':>{w}s}" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:>{w}.{dec}f}"
    print(f"{r[0]:<12s}{r[1]:<5s}{r[2]:<4s}" + f(r[3], 6) + f(r[4], 6) + f(r[5], 6, 2)
          + f(r[6], 7) + f(r[7], 7, 2) + f(r[8], 7, 2) + f(r[9], 7, 3) + f(r[10], 7, 3)
          + f(r[11], 6) + (f"{'--':>5s}" if np.isnan(r[12]) else f"{int(r[12]):>5d}"))
print("-" * len(hdr))
print("Window 2017-04-01 -> 2026-07-01 | daily bars, signals at close, fills next open")
print("slippage 0.02%/side | Ann/DD/Tr columns in % | Hold in trading days")
print("AvgDD/MedDD = avg/median depth of each drawdown episode (peak-to-recovery)")
