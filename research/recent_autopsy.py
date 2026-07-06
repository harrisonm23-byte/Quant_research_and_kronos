"""A. Autopsy: what separates 2022-2026 IBS winners from losers?
(the era where the GREEN grade decayed from ~76% to ~62%)
Rule: a condition only counts if its sign agrees on QQQ AND SPY.
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
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["volx"] = df["volume"] / df["volume"].rolling(20).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    df["range_x"] = (h - l) / tr.rolling(14).mean()
    df["rsi14"] = wilder_rsi(c)
    m20, s20 = c.rolling(20).mean(), c.rolling(20).std()
    df["pctb"] = (c - (m20 - 2 * s20)) / (4 * s20)
    df["sma200"] = c.rolling(200).mean()
    df["hi252"] = c.rolling(252).max()
    df["ret1"] = c.pct_change()
    df["ret10"] = c / c.shift(10) - 1
    df["rv20"] = np.log(c / c.shift(1)).rolling(20).std()
    df["rv80"] = df["rv20"].expanding().quantile(0.8)
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    df["prev_ibs_sig"] = (pd.Series(df["ibs"]).shift(1) < 0.20)
    df["next_open"] = o.shift(-1)
    df["gap_at_entry"] = df["next_open"] / c - 1     # known at fill time
    return df


sk = pd.read_csv(os.path.join(OUT, "SKEW_History.csv"))
sk["date"] = pd.to_datetime(sk["DATE"])
sk["chg5"] = sk["SKEW"] - sk["SKEW"].shift(5)
skmap = dict(zip(sk["date"], sk["chg5"]))

FEATURES = [
    ("gap-down >0.5% at entry open", lambda r: r.gap_at_entry < -0.005),
    ("below SMA200",                lambda r: r.close < r.sma200),
    (">10% off 52w high",           lambda r: r.close / r.hi252 - 1 < -0.10),
    ("crisis vol (rv>80th pct)",    lambda r: r.rv20 > r.rv80),
    ("SKEW rose >2 (5d)",           lambda r: skmap.get(r.date, 0) > 2),
    ("prev day also IBS<0.20",      lambda r: bool(r.prev_ibs_sig)),
    ("2wk decline >5%",             lambda r: r.ret10 < -0.05),
    ("below lower BB (%B<0)",       lambda r: r.pctb < 0),
    ("day was Friday",              lambda r: r.date.weekday() == 4),
]

res = {}
for sym in ["QQQ", "SPY"]:
    df = prep(sym)
    idx = {d: i for i, d in enumerate(df["date"])}
    tr = pd.read_csv(os.path.join(OUT, f"trades_S4_IBS_{sym}_e20_x70.csv"), parse_dates=["entry_date"])
    tr = tr[tr["entry_date"] >= "2022-01-01"]
    rows = list(df.itertuples(index=False))
    recs = []
    for t in tr.itertuples():
        i = idx.get(t.entry_date)
        if i and i > 0:
            recs.append((rows[i - 1], t.ret > 0))
    base = np.mean([w for _, w in recs])
    res[sym] = {"base": base, "n": len(recs), "f": {}}
    for name, fn in FEATURES:
        wt = [w for r, w in recs if fn(r)]
        wf = [w for r, w in recs if not fn(r)]
        if len(wt) >= 8 and len(wf) >= 8:
            res[sym]["f"][name] = (np.mean(wt) - np.mean(wf), len(wt))

print(f"2022-2026 IBS trades: QQQ n={res['QQQ']['n']} (WR {res['QQQ']['base']:.1%}), "
      f"SPY n={res['SPY']['n']} (WR {res['SPY']['base']:.1%})")
print(f"{'condition':<30s}{'QQQ dWR':>9s}{'SPY dWR':>9s}{'agree':>7s}{'nQ':>4s}")
print("-" * 60)
out = []
for name, _ in FEATURES:
    q = res["QQQ"]["f"].get(name)
    s = res["SPY"]["f"].get(name)
    if q is None or s is None:
        continue
    agree = (q[0] > 0) == (s[0] > 0)
    out.append((name, q[0], s[0], agree, q[1], min(abs(q[0]), abs(s[0])) if agree else 0))
out.sort(key=lambda x: -x[5])
for name, qg, sg, ag, nq, _ in out:
    print(f"{name:<30s}{qg:>+9.1%}{sg:>+9.1%}{'YES' if ag else 'no':>7s}{nq:>4d}")
