"""Does CBOE SKEW (index put-smirk) predict SPY's NEXT DAY?

1. Next-day SPY return by SKEW rolling-1y percentile quintile
   (rolling percentile because SKEW has secular upward drift).
2. Next-day return by 5-day SKEW change.
3. As context for IBS<0.20 signals: trade WR by SKEW state at signal.
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))

sk = pd.read_csv(os.path.join(OUT, "SKEW_History.csv"))
sk["date"] = pd.to_datetime(sk["DATE"])
sk = sk[["date", "SKEW"]].sort_values("date").reset_index(drop=True)
sk["pct1y"] = sk["SKEW"].rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True)
sk["chg5"] = sk["SKEW"] - sk["SKEW"].shift(5)

spy = pd.read_csv(os.path.join(OUT, "SPY_daily.csv"), parse_dates=["date"])
spy = spy.sort_values("date").reset_index(drop=True)
spy["next_ret"] = spy["close"].shift(-1) / spy["close"] - 1
df = spy.merge(sk, on="date", how="inner").dropna(subset=["next_ret", "pct1y", "chg5"])
print(f"merged days: {len(df)}  ({df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()})")
print(f"baseline next-day: P(up)={ (df['next_ret']>0).mean():.1%}  avg={df['next_ret'].mean():+.4%}\n")

print("--- 1. by SKEW rolling-1y percentile (level) ---")
df["q"] = pd.cut(df["pct1y"], [0, 20, 40, 60, 80, 100.01], labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
for q, g in df.groupby("q", observed=True):
    print(f"  {q:<8s} n={len(g):>4d}  P(up)={(g['next_ret']>0).mean():>6.1%}  avg={g['next_ret'].mean():>+8.4%}")

print("\n--- 2. by 5-day SKEW change ---")
df["c"] = pd.cut(df["chg5"], [-99, -6, -2, 2, 6, 99],
                 labels=["fell >6", "fell 2-6", "flat", "rose 2-6", "rose >6"])
for c, g in df.groupby("c", observed=True):
    print(f"  {c:<9s} n={len(g):>4d}  P(up)={(g['next_ret']>0).mean():>6.1%}  avg={g['next_ret'].mean():>+8.4%}")

print("\n--- 3. SKEW state at IBS<0.20 signal -> trade outcome ---")
skmap = dict(zip(sk["date"], zip(sk["pct1y"], sk["chg5"])))
for sym in ["QQQ", "SPY"]:
    tr = pd.read_csv(os.path.join(OUT, f"trades_S4_IBS_{sym}_e20_x70.csv"), parse_dates=["entry_date"])
    rows = []
    for t in tr.itertuples():
        sig_day = t.entry_date - pd.Timedelta(days=1)
        # walk back to the most recent SKEW print on/before signal day
        for back in range(0, 5):
            d = t.entry_date - pd.Timedelta(days=1 + back)
            if d in skmap and not np.isnan(skmap[d][0]):
                rows.append((skmap[d][0], skmap[d][1], t.ret))
                break
    a = pd.DataFrame(rows, columns=["pct", "chg", "ret"])
    hi = a[a["pct"] >= 66.7]
    lo = a[a["pct"] <= 33.3]
    up = a[a["chg"] > 2]
    dn = a[a["chg"] < -2]
    print(f"  {sym}: n={len(a)}  base WR={(a['ret']>0).mean():.1%}")
    print(f"    SKEW high (top tercile 1y): n={len(hi):>3d} WR={(hi['ret']>0).mean():>6.1%}  |  "
          f"low tercile: n={len(lo):>3d} WR={(lo['ret']>0).mean():>6.1%}")
    print(f"    SKEW rose >2 last 5d:       n={len(up):>3d} WR={(up['ret']>0).mean():>6.1%}  |  "
          f"fell >2: n={len(dn):>3d} WR={(dn['ret']>0).mean():>6.1%}")
