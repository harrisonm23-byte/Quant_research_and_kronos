"""Context dictionary: what conditions are present when IBS<0.20 signals WIN?

For every historical IBS trade on QQQ AND SPY, tag 12 context features at
signal time; report WR(feature) vs WR(~feature) per symbol; rank by agreement
(same sign on both symbols) and combined magnitude.
"""
import math
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def prep(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    c, h, l = df["close"], df["high"], df["low"]
    df["volx"] = df["volume"] / df["volume"].rolling(20).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    df["atr14"] = tr.rolling(14).mean()
    df["range_x"] = (h - l) / df["atr14"]
    df["rsi14"] = wilder_rsi(c)
    m20 = c.rolling(20).mean()
    s20 = c.rolling(20).std()
    df["pctb"] = (c - (m20 - 2 * s20)) / (4 * s20)
    df["sma20"] = m20
    df["sma50"] = c.rolling(50).mean()
    df["hi20"] = c.rolling(20).max()
    df["ret1"] = c.pct_change()
    df["dn3"] = (df["ret1"] < 0) & (df["ret1"].shift(1) < 0) & (df["ret1"].shift(2) < 0)
    df["ret10"] = c / c.shift(10) - 1
    df["rv20"] = np.log(c / c.shift(1)).rolling(20).std()
    df["rv_med"] = df["rv20"].expanding().median()
    df["weekday"] = df["date"].dt.weekday
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    return df


DFS = {s: prep(s) for s in ["QQQ", "SPY"]}
IDX = {s: {d: i for i, d in enumerate(DFS[s]["date"])} for s in DFS}

# cross-asset: symbol's day return minus the other symbol's day return
for s, other in [("QQQ", "SPY"), ("SPY", "QQQ")]:
    a, b = DFS[s], DFS[other]
    merged = a[["date", "ret1"]].merge(b[["date", "ret1"]], on="date", suffixes=("", "_o"))
    a["rel_ret"] = merged["ret1"] - merged["ret1_o"]

FEATURES = [
    ("quiet volume (<=1.2x)",   lambda r: r.volx <= 1.2),
    ("violent day (range>1.5xATR)", lambda r: r.range_x > 1.5),
    ("3+ consecutive red closes",   lambda r: bool(r.dn3)),
    ("downtrend (close<close-10d)", lambda r: r.ret10 < 0),
    ("shallow pullback (<3% off 20d-hi)", lambda r: r.close / r.hi20 - 1 > -0.03),
    ("RSI14 < 35",              lambda r: r.rsi14 < 35),
    ("below lower BB (%B<0)",   lambda r: r.pctb < 0),
    ("SMA20 > SMA50 (uptrend)", lambda r: r.sma20 > r.sma50),
    ("high-vol regime (rv>med)", lambda r: r.rv20 > r.rv_med),
    ("lagging other index >0.5%", lambda r: r.rel_ret < -0.005),
    ("signal on Friday",        lambda r: r.weekday == 4),
    ("deep IBS (<0.10)",        lambda r: r.ibs < 0.10),
]

res = {}
for sym in ["QQQ", "SPY"]:
    tr = pd.read_csv(os.path.join(OUT, f"trades_S4_IBS_{sym}_e20_x70.csv"), parse_dates=["entry_date"])
    df = DFS[sym]
    rows = list(df.itertuples(index=False))
    recs = []
    for t in tr.itertuples():
        i = IDX[sym].get(t.entry_date)
        if i is None or i == 0:
            continue
        recs.append((rows[i - 1], t.ret > 0))
    base = np.mean([w for _, w in recs])
    res[sym] = {"base": base, "n": len(recs), "feat": {}}
    for name, fn in FEATURES:
        wt = [w for r, w in recs if fn(r)]
        wf = [w for r, w in recs if not fn(r)]
        if len(wt) >= 15 and len(wf) >= 15:
            res[sym]["feat"][name] = (np.mean(wt) - np.mean(wf), np.mean(wt), len(wt))

print(f"IBS<0.20 trades — QQQ n={res['QQQ']['n']} (base WR {res['QQQ']['base']:.1%}), "
      f"SPY n={res['SPY']['n']} (base WR {res['SPY']['base']:.1%})\n")
hdr = f"{'condition at signal':<34s}{'QQQ dWR':>9s}{'SPY dWR':>9s}{'agree':>7s}{'nQ':>5s}"
print(hdr)
print("-" * len(hdr))
ranked = []
for name, _ in FEATURES:
    q = res["QQQ"]["feat"].get(name)
    s = res["SPY"]["feat"].get(name)
    if q is None or s is None:
        continue
    agree = (q[0] > 0) == (s[0] > 0)
    ranked.append((name, q[0], s[0], agree, q[2], min(abs(q[0]), abs(s[0])) if agree else 0))
ranked.sort(key=lambda x: -x[5])
for name, qg, sg, agree, nq, _ in ranked:
    print(f"{name:<34s}{qg:>+9.1%}{sg:>+9.1%}{'YES' if agree else 'no':>7s}{nq:>5d}")
print("\ndWR = WR(condition true) - WR(condition false). 'agree' = same sign both symbols.")
