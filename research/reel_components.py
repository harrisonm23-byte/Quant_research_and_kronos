"""Test the reel's two untested components.

1. BTC 1h momentum breakout: close > 24-bar high -> long next open,
   1% hard stop (per the video), exits: 2R target / 24-bar timeout.
   Variants: with/without the 1% stop; long-only (video implies long momentum).
2. GLD/USO trend-following (daily proxy for their '4h'): 10/40 MA cross,
   long on golden cross, exit on cross down; variants: long/short both;
   with the video's 1% stop vs none. Costs 0.02%/side (crypto real costs higher).
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
SLIP = 0.0002


def stats(rets, label, per_year=None):
    a = np.array(rets)
    if not len(a):
        print(f"{label:<34s} no trades")
        return
    wins, losses = a[a > 0], a[a <= 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else 99
    eq = np.cumprod(1 + a)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    tot = eq[-1] - 1
    print(f"{label:<34s} n={len(a):>4d}  WR={(a>0).mean():>6.1%}  avg={a.mean():>+7.3%}  "
          f"PF={pf:>5.2f}  total={tot:>+8.1%}  maxDD={mdd:>7.1%}")


# ---------- 1. BTC 1h momentum breakout ----------
b = pd.read_csv(os.path.join(OUT, "BTC_1h.csv"), parse_dates=["ts"]).sort_values("ts").reset_index(drop=True)
o, h, l, c = b["open"].values, b["high"].values, b["low"].values, b["close"].values
hi24 = pd.Series(h).shift(1).rolling(24).max().values
n = len(b)
print("=== BTC/USD 1h momentum breakout (close > prior 24-bar high) ===")
for use_stop, lbl in [(True, "1% stop, 2R target, 24-bar out"), (False, "no stop, exit @24 bars")]:
    rets = []
    i = 25
    while i < n - 26:
        if c[i] > hi24[i]:
            e = o[i + 1] * (1 + SLIP)
            stop = e * 0.99
            tgt = e * 1.02
            ret = None
            for k in range(i + 1, i + 25):
                if use_stop and l[k] <= stop:
                    ret = (stop * (1 - SLIP)) / e - 1
                    break
                if use_stop and h[k] >= tgt:
                    ret = (tgt * (1 - SLIP)) / e - 1
                    break
            if ret is None:
                ret = (c[min(i + 24, n - 1)] * (1 - SLIP)) / e - 1
            rets.append(ret)
            i += 24
        else:
            i += 1
    stats(rets, f"BTC long breakout, {lbl}")

# ---------- 2. GLD / USO trend following ----------
print("\n=== GLD / USO trend-following (10/40 MA cross, daily proxy for '4h') ===")
for sym in ["GLD", "USO"]:
    d = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"), parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    c_ = d["close"]
    d["f"] = c_.rolling(10).mean()
    d["s"] = c_.rolling(40).mean()
    o_, lo_, cl_ = d["open"].values, d["low"].values, c_.values
    f_, s_ = d["f"].values, d["s"].values
    for use_stop, lbl in [(False, "no stop"), (True, "video's 1% stop")]:
        rets = []
        in_pos = False
        e = None
        for i in range(41, len(d) - 1):
            if in_pos:
                if use_stop and lo_[i] <= e * 0.99:
                    rets.append((e * 0.99 * (1 - SLIP)) / e - 1)
                    in_pos = False
                    continue
                if f_[i] < s_[i]:
                    rets.append((o_[i + 1] * (1 - SLIP)) / e - 1)
                    in_pos = False
            else:
                if f_[i] > s_[i] and f_[i - 1] <= s_[i - 1]:
                    e = o_[i + 1] * (1 + SLIP)
                    in_pos = True
        stats(rets, f"{sym} long 10/40 cross, {lbl}")
    bh = cl_[-1] / cl_[40] - 1
    print(f"{'  ('+sym+' buy&hold same window)':<34s} total={bh:>+8.1%}")
