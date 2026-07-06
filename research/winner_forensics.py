"""Winner forensics: for each winning strategy's trades, which indicator
conditions at the signal close separate winners from losers?

For every feature F: WR(trades where F true) vs WR(trades where F false).
Signal day = session before the entry fill (fills are next-open).
"""
import math
import os

import numpy as np
import pandas as pd

from engine import load_symbol

OUT = os.path.dirname(os.path.abspath(__file__))


def wilder_rsi(close, period):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def features(sym):
    df = load_symbol(sym)
    c = df["close"]
    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["rsi14"] = wilder_rsi(c, 14)
    sd = c.rolling(20).std()
    df["pctb"] = (c - (df["sma20"] - 2 * sd)) / (4 * sd)
    df["vol20"] = df["volume"].rolling(20).mean()
    df["volx"] = df["volume"] / df["vol20"]
    df["rv20"] = np.log(c / c.shift(1)).rolling(20).std() * math.sqrt(252)
    df["rv20_med"] = df["rv20"].expanding().median()
    df["dist20"] = c / df["sma20"] - 1
    return df


FEATURES = [
    ("SMA5>SMA20", lambda r: r.sma5 > r.sma20),
    ("SMA20>SMA50", lambda r: r.sma20 > r.sma50),
    ("close>SMA200", lambda r: r.close > r.sma200),
    ("EMA9>EMA21", lambda r: r.ema9 > r.ema21),
    ("RSI14<30", lambda r: r.rsi14 < 30),
    ("RSI2<5", lambda r: r.rsi2 < 5),
    ("%B<0.05 (at low band)", lambda r: r.pctb < 0.05),
    ("vol>1.2x avg", lambda r: r.volx > 1.2),
    ("IBS<0.10", lambda r: r.ibs < 0.10),
    ("day ret<=-1%", lambda r: r.ret1 <= -0.01),
    (">2% below SMA20", lambda r: r.dist20 < -0.02),
    ("high-vol regime", lambda r: r.rv20 > r.rv20_med),
    ("signal on Mon", lambda r: r.weekday == 0),
    ("signal on Fri", lambda r: r.weekday == 4),
]

STRATS = [
    ("S9_5DayLow_A_QQQ", "QQQ"),
    ("S4_IBS_QQQ_e20_x70", "QQQ"),
    ("S1_DoubleSeven_QQQ", "QQQ"),
    ("S6_TT_A_QQQ", "QQQ"),
    ("S6_TT_C_QQQ", "QQQ"),
]

DFS = {s: features(s) for s in ["QQQ", "SPY"]}
IDX = {s: {d: i for i, d in enumerate(DFS[s]["date"])} for s in DFS}

agg = {}   # feature -> list of (wr_true, wr_false, n_true, n_false) across strategies
for run, sym in STRATS:
    tr = pd.read_csv(os.path.join(OUT, f"trades_{run}.csv"), parse_dates=["entry_date"])
    df = DFS[sym]
    rows = list(df.itertuples(index=False))
    print(f"\n=== {run}  ({len(tr)} trades, overall WR {(tr['ret']>0).mean():.1%}) ===")
    print(f"{'feature':<24s} {'WR|true':>8s} {'n':>5s} {'WR|false':>9s} {'n':>5s} {'gap':>7s}")
    recs = []
    for t in tr.itertuples():
        i = IDX[sym].get(t.entry_date)
        if i is None or i == 0:
            continue
        recs.append((rows[i - 1], t.ret > 0))   # signal day = day before fill
    for fname, fn in FEATURES:
        wt = [win for r, win in recs if fn(r)]
        wf = [win for r, win in recs if not fn(r)]
        if len(wt) < 10 or len(wf) < 10:
            continue
        wr_t, wr_f = np.mean(wt), np.mean(wf)
        gap = wr_t - wr_f
        agg.setdefault(fname, []).append(gap)
        mark = " <<" if abs(gap) >= 0.10 else ""
        print(f"{fname:<24s} {wr_t:>8.1%} {len(wt):>5d} {wr_f:>9.1%} {len(wf):>5d} {gap:>+7.1%}{mark}")

print("\n=== Cross-strategy consistency (avg WR gap, # strategies with same sign) ===")
for fname, gaps in sorted(agg.items(), key=lambda kv: -abs(np.mean(kv[1]))):
    same = max(sum(1 for g in gaps if g > 0), sum(1 for g in gaps if g < 0))
    print(f"{fname:<24s} avg gap {np.mean(gaps):+.1%}  ({same}/{len(gaps)} same sign)")
