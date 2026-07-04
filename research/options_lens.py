"""Options-overlay lens: fixed-horizon directional stats for high-WR signals.

For each selected run's entries (next-open fill incl. slippage), compute:
- P(close > entry) at 1, 2, 3, 5 sessions after entry day
- P(max favorable excursion >= +1%, +2%) within 3 sessions (intraday highs)
- median / mean return at each horizon
Also prints the full WR ranking (Wilson lower bound) across all runs.
"""
import glob
import math
import os

import numpy as np
import pandas as pd

from engine import load_symbol

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = {s: load_symbol(s) for s in ["SPY", "QQQ"]}
IDX = {s: {d: i for i, d in enumerate(DATA[s]["date"])} for s in DATA}


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    mg = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr - mg) / den


# ---- full WR ranking across all runs ----
rows = []
for path in sorted(glob.glob(os.path.join(OUT, "trades_*.csv"))):
    run = os.path.basename(path)[7:-4]
    tr = pd.read_csv(path)
    n = len(tr)
    k = int((tr["ret"] > 0).sum())
    rows.append((run, n, k / n if n else 0, wilson_lo(k, n)))
rows.sort(key=lambda r: -r[3])
print("=== Win-rate ranking (sorted by Wilson 95% lower bound) ===")
print(f"{'run':<28s} {'n':>5s} {'WR':>7s} {'WR_lo95':>8s}")
for run, n, wr, lo in rows[:15]:
    print(f"{run:<28s} {n:>5d} {wr:>7.1%} {lo:>8.1%}")

# ---- fixed-horizon options lens for selected runs ----
DEEP = [
    ("S1_DoubleSeven_QQQ", "QQQ"),
    ("S1_DoubleSeven_SPY", "SPY"),
    ("S7_TripleRSI_SPY", "SPY"),
    ("S6_TT_C_SPY", "SPY"),
    ("S6_TT_C_QQQ", "QQQ"),
    ("S5_LowerBand_QQQ_B_sma300", "QQQ"),
    ("S4_IBS_QQQ_e20_x70", "QQQ"),
    ("S9_5DayLow_A_QQQ", "QQQ"),
    ("S6_TT_A_QQQ", "QQQ"),
]
HORIZONS = [1, 2, 3, 5]

print("\n=== Fixed-horizon directional stats (from next-open entry fill) ===")
hdr = f"{'run':<28s} {'n':>4s} " + " ".join(f"{'P+' + str(h) + 'd':>6s}" for h in HORIZONS)
hdr += f" {'med3d':>7s} {'avg3d':>7s} {'MFE>=1%':>8s} {'MFE>=2%':>8s}"
print(hdr)
for run, sym in DEEP:
    path = os.path.join(OUT, f"trades_{run}.csv")
    if not os.path.exists(path):
        continue
    tr = pd.read_csv(path, parse_dates=["entry_date"])
    df = DATA[sym]
    closes = df["close"].values
    highs = df["high"].values
    n_bars = len(df)
    res = {h: [] for h in HORIZONS}
    mfe3 = []
    for t in tr.itertuples():
        i = IDX[sym].get(t.entry_date)
        if i is None:
            continue
        for h in HORIZONS:
            j = i + h - 1          # close of h-th session, entry day = session 1
            if j < n_bars:
                res[h].append(closes[j] / t.entry_px - 1)
        j3 = min(i + 2, n_bars - 1)
        mfe3.append(highs[i:j3 + 1].max() / t.entry_px - 1)
    n = len(mfe3)
    r3 = np.array(res[3])
    mfe3 = np.array(mfe3)
    line = f"{run:<28s} {n:>4d} "
    line += " ".join(f"{(np.array(res[h]) > 0).mean():>6.0%}" for h in HORIZONS)
    line += f" {np.median(r3):>+7.2%} {r3.mean():>+7.2%}"
    line += f" {(mfe3 >= 0.01).mean():>8.0%} {(mfe3 >= 0.02).mean():>8.0%}"
    print(line)

print("\nP+Nd = fraction of entries with close above entry fill N sessions later")
print("MFE  = max intraday high within 3 sessions of entry (spike-exit opportunity)")
