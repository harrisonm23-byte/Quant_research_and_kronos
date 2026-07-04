"""Gauntlet pass:
1. Gap-up continuation (gap>+0.5% -> long open, exit close): QQQ replication + time-split
2. Turn-of-month FIXED (first pass had a slicing bug)
3. Same-day gap-fill probability FIXED (full session, not first 30m)
4. Grade v2 (v1 + Friday +1 + SKEW-rising +0.5) time-split vs v1
"""
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    mg = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr - mg) / den


HALVES = [("2016-2021", "2016-06-01", "2022-01-01"), ("2022-2026", "2022-01-01", "2027-01-01")]

print("=== 1. GAP-UP CONTINUATION (gap>+0.5%, long open -> close, 0.02%/side) ===")
print(f"{'symbol/half':<18s}{'n':>5s}{'WR':>8s}{'WR_lo95':>9s}{'avg':>9s}")
for sym in ["SPY", "QQQ"]:
    d = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"), parse_dates=["date"])
    d = d.sort_values("date").reset_index(drop=True)
    d["gap"] = d["open"] / d["close"].shift(1) - 1
    d["oc"] = d["close"] * (1 - 0.0002) / (d["open"] * (1 + 0.0002)) - 1
    for label, lo_d, hi_d in HALVES:
        g = d[(d["date"] >= lo_d) & (d["date"] < hi_d) & (d["gap"] > 0.005)]
        a = g["oc"].values
        k = (a > 0).sum()
        print(f"{sym+' '+label:<18s}{len(a):>5d}{k/len(a):>8.1%}{wilson_lo(k,len(a)):>9.1%}{a.mean():>+9.3%}")

print("\n=== 2. TURN-OF-MONTH (fixed) ===")
for sym in ["SPY", "QQQ"]:
    d = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"), parse_dates=["date"])
    d = d[d["date"] >= "2016-06-01"].sort_values("date").reset_index(drop=True)
    d["ret"] = d["close"].pct_change()
    d["tom"] = False
    for m, g in d.groupby(d["date"].dt.to_period("M")):
        pos = g.index.tolist()
        for i in pos[:3] + pos[-2:]:
            d.loc[i, "tom"] = True
    tom, rest = d.loc[d["tom"], "ret"].dropna(), d.loc[~d["tom"], "ret"].dropna()
    print(f"  {sym}: ToM avg {tom.mean():+.4%}/day (n={len(tom)}, P(up)={(tom>0).mean():.1%})  "
          f"vs rest {rest.mean():+.4%}/day (n={len(rest)}, P(up)={(rest>0).mean():.1%})")

print("\n=== 3. SAME-DAY GAP FILL (fixed: full session) ===")
d = pd.read_csv(os.path.join(OUT, "SPY_daily.csv"), parse_dates=["date"])
d = d[d["date"] >= "2016-06-01"].sort_values("date").reset_index(drop=True)
d["gap"] = d["open"] / d["close"].shift(1) - 1
d["pc"] = d["close"].shift(1)
d = d.dropna()
for lo_b, hi_b, lbl in [(-9, -0.005, "gap<-0.5%"), (-0.005, -0.001, "-0.5..-0.1"),
                        (0.001, 0.005, "+0.1..0.5"), (0.005, 9, "gap>+0.5%")]:
    g = d[(d["gap"] >= lo_b) & (d["gap"] < hi_b)]
    if lo_b >= 0:
        filled = (g["low"] <= g["pc"]).mean()
    else:
        filled = (g["high"] >= g["pc"]).mean()
    print(f"  {lbl:<11s} n={len(g):>4d}  P(fill same day)={filled:.1%}")

print("\n=== 4. GRADE v2 vs v1 time-split (IBS trades) ===")
import context_dictionary as cd  # noqa: E402  (prints its own table once)
sk = pd.read_csv(os.path.join(OUT, "SKEW_History.csv"))
sk["date"] = pd.to_datetime(sk["DATE"])
sk["chg5"] = sk["SKEW"] - sk["SKEW"].shift(5)
skmap = dict(zip(sk["date"], sk["chg5"]))


def g1(r):
    return ((r.volx <= 1.2) + (r.sma20 > r.sma50)
            - (r.rsi14 < 35) - bool(r.dn3) - (r.range_x > 1.5))


def g2(r):
    sk5 = skmap.get(r.date, 0)
    return g1(r) + (r.weekday == 4) + 0.5 * (sk5 > 2 if not np.isnan(sk5) else 0)


print(f"{'cell':<26s}{'n':>5s}{'WR':>8s}{'WR_lo95':>9s}{'avg':>9s}")
for sym in ["QQQ", "SPY"]:
    tr = pd.read_csv(os.path.join(OUT, f"trades_S4_IBS_{sym}_e20_x70.csv"), parse_dates=["entry_date"])
    df = cd.DFS[sym]
    rows = list(df.itertuples(index=False))
    recs = [(rows[cd.IDX[sym][t.entry_date] - 1], t.ret, t.entry_date)
            for t in tr.itertuples() if cd.IDX[sym].get(t.entry_date, 0) > 0]
    for label, lo_d, hi_d in HALVES:
        sel = [ret for r, ret, dte in recs
               if pd.Timestamp(lo_d) <= dte < pd.Timestamp(hi_d) and g2(r) >= 2.5]
        a = np.array(sel)
        if len(a):
            k = (a > 0).sum()
            print(f"{sym+' v2>=2.5 '+label:<26s}{len(a):>5d}{k/len(a):>8.1%}"
                  f"{wilson_lo(k, len(a)):>9.1%}{a.mean():>+9.3%}")
