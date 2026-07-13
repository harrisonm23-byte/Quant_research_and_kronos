"""Scan for quick-direction / high-WR signals suited to OTM call scalping.

Scores each candidate on:
  - P(underlying +0.75% intraday within 2 sessions from entry open)
  - P(underlying +1.0% within 2 sessions)
  - P(close > entry at 1d and 2d)
  - Underlying trade WR (if defined exit) or fixed-horizon WR at 2d
  - Wilson 95% lower bound on 2d-horizon win rate
  - Time-split stability (2017-2021 vs 2022-2026)

Outputs top candidates and replicates the best as a formal spec with OTM sim.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

from engine import STAT_START, load_symbol, run_bt

OUT = os.path.dirname(os.path.abspath(__file__))
HALVES = [("H1", "2017-04-01", "2022-01-01"), ("H2", "2022-01-01", "2027-01-01")]


def wilson_lo(k, n, z=1.96):
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    mg = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (ctr - mg) / den


def _nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def prep(sym):
    df = load_symbol(sym)
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["volx"] = v / v.rolling(20).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    df["atr14"] = tr.rolling(14).mean()
    df["range_x"] = (h - l) / df["atr14"]
    df["rsi14"] = wilder_rsi(c)
    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    df["ret1"] = c.pct_change()
    df["dn3"] = (df["ret1"] < 0) & (df["ret1"].shift(1) < 0) & (df["ret1"].shift(2) < 0)
    df["weekday"] = df["date"].dt.weekday
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    df["lc5"] = c.rolling(5).min()
    df["ll5"] = l.rolling(5).min()
    df["lc7"] = c.rolling(7).min()
    df["hc7"] = c.rolling(7).max()
    df["prev_close"] = c.shift(1)
    df["prev_low"] = l.shift(1)
    df["prev2_close"] = c.shift(2)
    df["hi20"] = c.rolling(20).max()
    hh10 = h.rolling(10).max()
    df["lower_band"] = hh10 - 2.5 * (h - l).rolling(25).mean()
    return df


def load_skew():
    path = os.path.join(OUT, "SKEW_History.csv")
    if not os.path.exists(path):
        return {}
    sk = pd.read_csv(path)
    sk["date"] = pd.to_datetime(sk["DATE"])
    sk["chg5"] = sk["SKEW"] - sk["SKEW"].shift(5)
    return dict(zip(sk["date"], sk["chg5"]))


def grade_v1(r):
    return ((r.volx <= 1.2) + (r.sma20 > r.sma50)
            - (r.rsi14 < 35) - bool(r.dn3) - (r.range_x > 1.5))


def grade_v2(r, sk):
    sk5 = sk.get(r.date, 0)
    return grade_v1(r) + (r.weekday == 4) + (0.5 if (not np.isnan(sk5) and sk5 > 2) else 0)


# Base entry definitions (signal at close t, enter open t+1)
CANDIDATES = [
    ("5DayLow_A", lambda r: _nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5),
    ("5DayLow_deepIBS", lambda r: _nn(r.lc5) and r.ibs < 0.20 and r.close <= r.lc5),
    ("5DayLow_quiet", lambda r: _nn(r.lc5, r.volx) and r.ibs < 0.25 and r.close <= r.lc5 and r.volx <= 1.2),
    ("5DayLow_green", lambda r: _nn(r.lc5, r.sma20, r.sma50) and r.ibs < 0.25 and r.close <= r.lc5
     and r.volx <= 1.2 and r.sma20 > r.sma50),
    ("5DayLow_Friday", lambda r: _nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5 and r.weekday == 4),
    ("IBS_20", lambda r: r.ibs < 0.20),
    ("IBS_15", lambda r: r.ibs < 0.15),
    ("IBS_10", lambda r: r.ibs < 0.10),
    ("IBS_green", lambda r: r.ibs < 0.20 and _nn(r.volx, r.sma20, r.sma50)
     and r.volx <= 1.2 and r.sma20 > r.sma50 and r.rsi14 >= 35 and not bool(r.dn3)),
    ("TT_A", lambda r: r.weekday == 0 and r.close < r.open),
    ("TT_B", lambda r: r.weekday == 0 and _nn(r.prev_low) and r.close < r.prev_low),
    ("TT_C", lambda r: (r.weekday == 0 and _nn(r.prev_close, r.prev2_close)
                        and r.close < r.prev_close and r.prev_close < r.prev2_close)),
    ("LowerBand_IBS", lambda r: _nn(r.lower_band) and r.close < r.lower_band and r.ibs < 0.30),
    ("D7_uptrend", lambda r: _nn(r.sma200, r.lc7) and r.close > r.sma200 and r.close <= r.lc7),
]


def scalp_metrics(df, entry_fn, post_filter=None):
    """Score entries for OTM scalp suitability (no ETF exit — fixed 2d horizon)."""
    rows = list(df.itertuples(index=False))
    idx = {d: i for i, d in enumerate(df["date"])}
    o, h, c = df["open"].values, df["high"].values, df["close"].values
    nb = len(df)
    start = pd.Timestamp(STAT_START)

    recs = []
    for i, r in enumerate(rows):
        if r.date < start or i >= nb - 3:
            continue
        if not entry_fn(r):
            continue
        if post_filter and not post_filter(r):
            continue
        ep = o[i + 1] * (1 + 0.0002)  # next open fill
        entry_date = df["date"].iloc[i + 1]
        w = h[i + 1:min(i + 3, nb)]
        hit75 = (w.max() / ep - 1) >= 0.0075 if len(w) else False
        hit100 = (w.max() / ep - 1) >= 0.010 if len(w) else False
        j1 = min(i + 1, nb - 1)
        j2 = min(i + 2, nb - 1)
        win1 = c[j1] > ep
        win2 = c[j2] > ep
        ret2 = c[j2] / ep - 1
        recs.append(dict(entry_date=entry_date, hit75=hit75, hit100=hit100,
                         win1=win1, win2=win2, ret2=ret2))

    if len(recs) < 15:
        return None
    tr = pd.DataFrame(recs)
    n = len(tr)
    k2 = int(tr["win2"].sum())
    return dict(
        n=n,
        p75=tr["hit75"].mean(),
        p100=tr["hit100"].mean(),
        p1=tr["win1"].mean(),
        p2=tr["win2"].mean(),
        wr2_lo=wilson_lo(k2, n),
        avg2=tr["ret2"].mean(),
        med2=tr["ret2"].median(),
        trades=tr,
    )


def time_split(tr, col="win2"):
    out = {}
    for label, lo, hi in HALVES:
        sub = tr[(tr["entry_date"] >= lo) & (tr["entry_date"] < hi)]
        if len(sub) >= 8:
            out[label] = sub[col].mean()
        else:
            out[label] = float("nan")
    return out


def score_row(m):
    """Composite: prioritize 2d WR lower bound, spike reach, sample size."""
    if m is None:
        return -999
    freq = min(m["n"] / 150.0, 1.0)  # prefer enough trades
    return (m["wr2_lo"] * 2.0 + m["p75"] * 1.5 + m["p2"] * 0.5
            + freq * 0.15 + m["avg2"] * 5.0)


def scan(sym="QQQ"):
    df = prep(sym)
    sk = load_skew()

    # Also test IBS + grade v2 post-filters on IBS_20 base entries
    results = []
    for name, fn in CANDIDATES:
        m = scalp_metrics(df, fn)
        if m:
            ts = time_split(m["trades"])
            results.append((name, "base", m, ts))

    # Grade v2 filtered IBS entries
    def ibs_g2(r):
        return r.ibs < 0.20 and grade_v2(r, sk) >= 2.5
    m = scalp_metrics(df, ibs_g2)
    if m:
        results.append(("IBS_grade2_A", "base", m, time_split(m["trades"])))

    def ibs_g1_green(r):
        return r.ibs < 0.20 and grade_v1(r) >= 2
    m = scalp_metrics(df, ibs_g1_green)
    if m:
        results.append(("IBS_green_v1", "base", m, time_split(m["trades"])))

    results.sort(key=lambda x: -score_row(x[2]))
    return results


def print_scan(results):
    print(f"\n{'='*90}")
    print("OTM SCALP SIGNAL SCAN — QQQ (enter next open, score 2-session horizon)")
    print(f"{'='*90}")
    hdr = (f"{'rank':<4s}{'signal':<22s}{'n':>4s}{'P+75':>6s}{'P+1%':>6s}{'P+1d':>6s}"
           f"{'P+2d':>6s}{'WR2lo':>6s}{'avg2d':>7s}{'H1_2d':>6s}{'H2_2d':>6s}{'score':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for rank, (name, _, m, ts) in enumerate(results[:15], 1):
        print(f"{rank:<4d}{name:<22s}{m['n']:>4d}{m['p75']:>6.0%}{m['p100']:>6.0%}"
              f"{m['p1']:>6.0%}{m['p2']:>6.0%}{m['wr2_lo']:>6.0%}{m['avg2']:>+7.2%}"
              f"{ts.get('H1', float('nan')):>6.0%}{ts.get('H2', float('nan')):>6.0%}"
              f"{score_row(m):>6.2f}")


# ---- OTM sim (compact) ----
def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, iv, kind):
    if T <= 0:
        return max(S - K, 0.0) if kind == "c" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (0.04 + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "c":
        return S * _N(d1) - K * math.exp(-0.04 * T) * _N(d2)
    return K * math.exp(-0.04 * T) * _N(-d2) - S * _N(-d1)


def otm_sim(df, trades, target=0.0075, max_sess=2, otm=0.02, expiry=5):
    rv = np.log(df["close"] / df["close"].shift(1)).rolling(20).std() * math.sqrt(252)
    idx = {d: i for i, d in enumerate(df["date"])}
    o, h, c = df["open"].values, df["high"].values, df["close"].values
    pnls = []
    for _, row in trades.iterrows():
        i = idx.get(row["entry_date"])
        if i is None or math.isnan(rv.iloc[i]):
            continue
        S0 = o[i]
        iv0 = rv.iloc[i]
        K = S0 * (1 + otm)
        spike = None
        for k in range(i, min(i + max_sess, len(h))):
            if h[k] >= S0 * (1 + target):
                spike = k
                break
        if spike is not None:
            Sx, Tx = S0 * (1 + target), max(expiry - (spike - i) - 0.5, 0.05) / 252.0
            ivx = iv0 * min(1.4, max(0.6, 1 - 3 * target))
        else:
            j = min(i + max_sess - 1, len(c) - 1)
            Sx = c[j]
            ret = Sx / S0 - 1
            Tx = max(expiry - max_sess, 0.05) / 252.0
            ivx = iv0 * min(1.4, max(0.6, 1 - 3 * ret))
        c0 = bs(S0, K, expiry / 252.0, iv0, "c") * 1.01
        c1 = bs(Sx, K, Tx, ivx, "c") * 0.99
        if c0 > 0:
            pnls.append((c1 - c0) / c0)
    a = np.array(pnls)
    if not len(a):
        return None
    return dict(n=len(a), wr=(a > 0).mean(), avg=a.mean(), med=np.median(a))


def replicate_winner(name, entry_fn, sym="QQQ"):
    df = prep(sym)
    m = scalp_metrics(df, entry_fn)
    if not m:
        print("No trades for winner")
        return
    print(f"\n{'='*90}")
    print(f"REPLICATED SPEC: {name} on {sym}")
    print(f"{'='*90}")
    print(f"""
RULES (OTM scalp sleeve):
  Signal bar  : daily QQQ, evaluated at close
  Entry       : next session open (+ 0.02% slippage)
  ETF exit    : N/A for scalp sleeve — options-only expression
  Options     : buy 2% OTM call (~5d expiry), 2 contracts
  Exit        : sell when QQQ intraday high >= entry +0.75%
                OR at close of session 2 if target not hit (theta stop)
  Skip if     : QQQ close < SMA200 (bear regime gate — recommended)
""")
    print("Underlying 2-session stats:")
    print(f"  n={m['n']}  P(+0.75% in 2d)={m['p75']:.1%}  P(+1.0% in 2d)={m['p100']:.1%}")
    print(f"  P(close>entry @1d)={m['p1']:.1%}  P(@2d)={m['p2']:.1%}  "
          f"WR2 Wilson-lo={m['wr2_lo']:.1%}")
    print(f"  avg 2d return={m['avg2']:+.3%}  median={m['med2']:+.3%}")
    ts = time_split(m["trades"])
    print(f"  time-split P+2d: H1={ts.get('H1', 0):.1%}  H2={ts.get('H2', 0):.1%}")

    # regime gate check
    rows = list(df.itertuples(index=False))
    idx = {d: i for i, d in enumerate(df["date"])}
    bull = []
    for _, row in m["trades"].iterrows():
        i = idx.get(row["entry_date"])
        if i and i > 0 and rows[i - 1].close > rows[i - 1].sma200:
            bull.append(row["win2"])
    if bull:
        print(f"  above SMA200 only: n={len(bull)} P+2d={np.mean(bull):.1%}")

    for tgt, lbl in [(0.0075, "+0.75%"), (0.010, "+1.0%")]:
        sim = otm_sim(df, m["trades"], target=tgt)
        if sim:
            print(f"  OTM sim exit {lbl}: n={sim['n']} WR={sim['wr']:.0%} "
                  f"avg={sim['avg']:+.0%} med={sim['med']:+.0%}")

    # persist trade log
    out_path = os.path.join(OUT, f"otm_scalp_{name}_{sym}.csv")
    m["trades"].to_csv(out_path, index=False)
    print(f"\n  Trade log -> {out_path}")


def main():
    results = scan("QQQ")
    print_scan(results)

    # replicate top candidates
    print("\n--- Promotion criteria for OTM scalp ---")
    print("  WR2 Wilson-lo >= 60%  |  P(+0.75% in 2d) >= 65%  |  n >= 30"
          "  |  both halves P+2d >= 55%")
    promoted = []
    for name, _, m, ts in results:
        h1, h2 = ts.get("H1", 0), ts.get("H2", 0)
        ok = (m["wr2_lo"] >= 0.60 and m["p75"] >= 0.65 and m["n"] >= 30
              and (np.isnan(h1) or h1 >= 0.55) and (np.isnan(h2) or h2 >= 0.55))
        if ok:
            promoted.append(name)
    print(f"  Promoted: {promoted if promoted else 'NONE — relaxing to top-1 for replication'}")

    # Map name -> entry fn
    fn_map = {n: f for n, f in CANDIDATES}
    fn_map["IBS_grade2_A"] = lambda r: r.ibs < 0.20 and grade_v2(r, load_skew()) >= 2.5
    fn_map["IBS_green_v1"] = lambda r: r.ibs < 0.20 and grade_v1(r) >= 2

    winner_name = promoted[0] if promoted else results[0][0]
    replicate_winner(winner_name, fn_map[winner_name])

    if len(results) > 1 and results[1][0] != winner_name:
        replicate_winner(results[1][0], fn_map.get(results[1][0], CANDIDATES[0][1]))


if __name__ == "__main__":
    main()
