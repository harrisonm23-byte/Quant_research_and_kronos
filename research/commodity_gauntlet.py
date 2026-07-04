"""Commodity trend-following gauntlet: GLD, USO, SLV, UNG.
- MA-cross long/flat, parameter sweep: (5,20),(10,40),(20,50),(20,100),(50,200)
- Time-split 2016-2021 / 2022-2026 for the 10/40 base case
- USO 2020-episode check: performance excluding Feb-Jun 2020
Costs 0.02%/side.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
SLIP = 0.0002


def run(d, fast, slow, start=None, end=None, skip=None):
    d = d.copy()
    c = d["close"]
    d["f"] = c.rolling(fast).mean()
    d["s"] = c.rolling(slow).mean()
    o_, cl_, f_, s_, dt = d["open"].values, c.values, d["f"].values, d["s"].values, d["date"].values
    rets, in_pos, e = [], False, None
    for i in range(slow + 1, len(d) - 1):
        if start is not None and dt[i] < np.datetime64(start):
            continue
        if end is not None and dt[i] >= np.datetime64(end):
            break
        if skip and np.datetime64(skip[0]) <= dt[i] < np.datetime64(skip[1]):
            if in_pos:
                rets.append((o_[i + 1] * (1 - SLIP)) / e - 1)
                in_pos = False
            continue
        if in_pos:
            if f_[i] < s_[i]:
                rets.append((o_[i + 1] * (1 - SLIP)) / e - 1)
                in_pos = False
        elif f_[i] > s_[i] and f_[i - 1] <= s_[i - 1]:
            e = o_[i + 1] * (1 + SLIP)
            in_pos = True
    return np.array(rets)


def line(rets):
    if not len(rets):
        return "no trades"
    eq = np.cumprod(1 + rets)
    wins, losses = rets[rets > 0], rets[rets <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
    return f"n={len(rets):>3d} PF={pf:>5.2f} total={eq[-1]-1:>+8.1%}"


DATA = {s: pd.read_csv(os.path.join(OUT, f"{s}_daily.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
        for s in ["GLD", "USO", "SLV", "UNG"]}

print("=== parameter sweep (full window, long/flat) ===")
print(f"{'pair':<10s}" + "".join(f"{s:>32s}" for s in DATA))
for fast, slow in [(5, 20), (10, 40), (20, 50), (20, 100), (50, 200)]:
    row = f"{f'{fast}/{slow}':<10s}"
    for s, d in DATA.items():
        row += f"{line(run(d, fast, slow)):>32s}"
    print(row)

print("\n=== time-split, 10/40 ===")
for s, d in DATA.items():
    a = run(d, 10, 40, end="2022-01-01")
    b = run(d, 10, 40, start="2022-01-01")
    print(f"{s}: 2016-2021 {line(a)}   |   2022-2026 {line(b)}")

print("\n=== USO 10/40 excluding Feb-Jun 2020 (the roll-catastrophe episode) ===")
print("USO ex-2020:", line(run(DATA["USO"], 10, 40, skip=("2020-02-01", "2020-07-01"))))
print("USO full   :", line(run(DATA["USO"], 10, 40)))

print("\n=== buy&hold reference (full window) ===")
for s, d in DATA.items():
    print(f"{s}: {d['close'].iloc[-1]/d['close'].iloc[40]-1:+.1%}")
