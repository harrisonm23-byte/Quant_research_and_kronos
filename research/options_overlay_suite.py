"""Options overlay suite: confirmation-speed ranking + put-spread simulation.

Runs short put spread (ATM / -2% OTM) on keeper and candidate signals using the
same Black-Scholes model as options_sim.py. Also scores Grade v2 filtered IBS
and prints a bear-regime / SQQQ sanity check.

Usage:
  python3 options_overlay_suite.py
"""
import math
import os

import numpy as np
import pandas as pd

from engine import STAT_START, load_symbol, run_bt

OUT = os.path.dirname(os.path.abspath(__file__))
R = 0.04
COST = 0.02
SLIP = 0.0002

# Grade v2 helpers (gauntlet_pass.py)
def _wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def prep_features(sym):
    df = load_symbol(sym)
    c, h, l = df["close"], df["high"], df["low"]
    df["volx"] = df["volume"] / df["volume"].rolling(20).mean()
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    df["atr14"] = tr.rolling(14).mean()
    df["range_x"] = (h - l) / df["atr14"]
    df["rsi14"] = _wilder_rsi(c)
    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["ret1"] = c.pct_change()
    df["dn3"] = (df["ret1"] < 0) & (df["ret1"].shift(1) < 0) & (df["ret1"].shift(2) < 0)
    df["weekday"] = df["date"].dt.weekday
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    df["lc7"] = c.rolling(7).min()
    df["hc7"] = c.rolling(7).max()
    df["lc5"] = c.rolling(5).min()
    df["prev_close"] = c.shift(1)
    df["sma200"] = c.rolling(200).mean()
    return df


def load_skew():
    path = os.path.join(OUT, "SKEW_History.csv")
    if not os.path.exists(path):
        return {}
    sk = pd.read_csv(path)
    sk["date"] = pd.to_datetime(sk["DATE"])
    sk["chg5"] = sk["SKEW"] - sk["SKEW"].shift(5)
    return dict(zip(sk["date"], sk["chg5"]))


def grade_v1(row):
    return ((row.volx <= 1.2) + (row.sma20 > row.sma50)
            - (row.rsi14 < 35) - bool(row.dn3) - (row.range_x > 1.5))


def grade_v2(row, skmap):
    sk5 = skmap.get(row.date, 0)
    extra = (row.weekday == 4) + (0.5 if (not np.isnan(sk5) and sk5 > 2) else 0)
    return grade_v1(row) + extra


def _nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


SIGNALS = {
    "IBS_QQQ": ("QQQ", dict(
        entry_fn=lambda r: r.ibs < 0.20,
        exit_fn=lambda r: r.ibs > 0.70)),
    "IBS_G2_QQQ": ("QQQ", dict(
        entry_fn=lambda r: r.ibs < 0.20,  # filtered post-hoc
        exit_fn=lambda r: r.ibs > 0.70)),
    "DoubleSeven_QQQ": ("QQQ", dict(
        entry_fn=lambda r: _nn(r.sma200, r.lc7) and r.close > r.sma200 and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7)),
    "5DayLow_A_QQQ": ("QQQ", dict(
        entry_fn=lambda r: _nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5,
        exit_fn=lambda r: _nn(r.prev_close) and r.close > r.prev_close)),
    "TT_A_QQQ": ("QQQ", dict(
        entry_fn=lambda r: r.weekday == 0 and r.close < r.open,
        exit_fn=None, max_hold=1)),
}


def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs(S, K, T, iv, kind):
    if T <= 0:
        return max(S - K, 0.0) if kind == "c" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (R + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "c":
        return S * _N(d1) - K * math.exp(-R * T) * _N(d2)
    return K * math.exp(-R * T) * _N(-d2) - S * _N(-d1)


def _iv_exit(iv0, ret):
    return iv0 * min(1.4, max(0.6, 1 - 3 * ret))


def confirmation_stats(df, trades):
    idx = {d: i for i, d in enumerate(df["date"])}
    closes, highs = df["close"].values, df["high"].values
    nb = len(df)
    p1, p2, p3, mfe1, first_pos = [], [], [], [], []
    for t in trades.itertuples():
        i = idx.get(t.entry_date)
        if i is None:
            continue
        ep = t.entry_px
        for hzn, bucket in [(1, p1), (2, p2), (3, p3)]:
            j = i + hzn - 1
            if j < nb:
                bucket.append(closes[j] > ep)
        j3 = min(i + 2, nb - 1)
        mfe1.append(highs[i:j3 + 1].max() / ep - 1 >= 0.01)
        fp = None
        for k in range(i, min(i + 5, nb)):
            if closes[k] > ep:
                fp = k - i + 1
                break
        if fp is not None:
            first_pos.append(fp)
    return dict(
        n=len(mfe1),
        p1=np.mean(p1) if p1 else 0,
        p2=np.mean(p2) if p2 else 0,
        p3=np.mean(p3) if p3 else 0,
        mfe1=np.mean(mfe1) if mfe1 else 0,
        med_first=np.median(first_pos) if first_pos else float("nan"),
        med_hold=trades["hold_days"].median() if len(trades) else float("nan"),
    )


def put_spread_sim(df, trades, horizon=None, expiry_pad=2, iv_mult=1.0):
    rv = np.log(df["close"] / df["close"].shift(1)).rolling(20).std() * math.sqrt(252)
    idx = {d: i for i, d in enumerate(df["date"])}
    o, c = df["open"].values, df["close"].values
    nb = len(df)
    if horizon is None:
        horizon = max(1, int(round(trades["hold_days"].median())))
    expiry = max(horizon + expiry_pad, 5)
    pnls = []
    for t in trades.itertuples():
        i = idx.get(t.entry_date)
        if i is None or math.isnan(rv.iloc[i]):
            continue
        j = min(i + horizon - 1, nb - 1)
        S0, S1 = o[i], c[j]
        iv0 = rv.iloc[i] * iv_mult
        K, K2 = S0, 0.98 * S0
        T0, T1 = expiry / 252.0, max(expiry - horizon, 0) / 252.0
        ret = S1 / S0 - 1
        ive = _iv_exit(iv0, ret)
        p0s, p0l = _bs(S0, K, T0, iv0, "p"), _bs(S0, K2, T0, iv0, "p")
        credit = (p0s - p0l) * (1 - COST / 2)
        p1s, p1l = _bs(S1, K, T1, ive, "p"), _bs(S1, K2, T1, ive, "p")
        cost_close = (p1s - p1l) * (1 + COST / 2)
        max_risk = (K - K2) - credit
        if max_risk <= 0:
            continue
        pnls.append((credit - cost_close) / max_risk)
    a = np.array(pnls)
    if not len(a):
        return None
    return dict(n=len(a), wr=(a > 0).mean(), avg=a.mean(), med=np.median(a),
                p25=np.percentile(a, 25), p75=np.percentile(a, 75), sum=a.sum())


def bear_regime_check():
    """Sanity: mean-reversion keepers in bear vs bull; SQQQ long attempts."""
    print("\n=== BEAR REGIME CHECK ===")
    qqq = load_symbol("QQQ")
    c = qqq["close"]
    qqq["sma200"] = c.rolling(200).mean()
    bull = qqq["close"] > qqq["sma200"]
    regimes = [("bull (close>SMA200)", bull), ("bear (close<SMA200)", ~bull)]

    keeper_kw = SIGNALS["IBS_QQQ"][1]
    eq, tr = run_bt(qqq, **keeper_kw)
    idx = {d: i for i, d in enumerate(qqq["date"])}
    for label, mask in regimes:
        sub = tr[tr["entry_date"].map(lambda d: mask.iloc[idx[d]] if d in idx else False)]
        if len(sub) < 10:
            print(f"  {label}: n={len(sub)} (too few)")
            continue
        wr = (sub["ret"] > 0).mean()
        avg = sub["ret"].mean()
        print(f"  IBS QQQ in {label}: n={len(sub):>3d}  WR={wr:.1%}  avg={avg:+.3%}")

    print("\n  SQQQ long mean-reversion (why we avoid inverse ETFs):")
    try:
        sqqq = load_symbol("SQQQ")
    except FileNotFoundError:
        print("    (no SQQQ_daily.csv — skip)")
        return
    for name, kw in [("IBS", SIGNALS["IBS_QQQ"][1]), ("5DayLow", SIGNALS["5DayLow_A_QQQ"][1])]:
        _, tr2 = run_bt(sqqq, **kw)
        if not len(tr2):
            continue
        print(f"    {name} on SQQQ: n={len(tr2)} WR={(tr2['ret']>0).mean():.1%} "
              f"avg={tr2['ret'].mean():+.3%} med_hold={tr2['hold_days'].median():.0f}d")


def tqqq_sleeve_check():
    print("\n=== TQQQ SLEEVE (1/3 notional on IBS + Grade v2 filter) ===")
    try:
        tqqq = prep_features("TQQQ")
    except FileNotFoundError:
        print("  (no TQQQ_daily.csv — skip)")
        return
    skmap = load_skew()
    rows = list(tqqq.itertuples(index=False))
    idx = {d: i for i, d in enumerate(tqqq["date"])}

    # IBS all vs grade v2 >= 2.5
    eq, tr = run_bt(tqqq, **SIGNALS["IBS_QQQ"][1])
    g2_dates = set()
    for t in tr.itertuples():
        i = idx.get(t.entry_date)
        if i and i > 0:
            r = rows[i - 1]
            if grade_v2(r, skmap) >= 2.5:
                g2_dates.add(t.entry_date)
    tr_g2 = tr[tr["entry_date"].isin(g2_dates)]
    for label, sub in [("IBS all", tr), ("IBS grade>=2.5", tr_g2)]:
        if not len(sub):
            continue
        scaled = sub["ret"].values / 3.0
        eqc = np.cumprod(1 + scaled)
        days = (tqqq["date"].iloc[-1] - pd.Timestamp(STAT_START)).days
        ann = eqc[-1] ** (365.25 / days) - 1
        peak = np.maximum.accumulate(eqc)
        mdd = (eqc / peak - 1).min()
        print(f"  {label}: n={len(sub):>3d}  WR={(scaled>0).mean():.1%}  "
              f"ann@1/3={ann:.1%}  maxDD@1/3={mdd:.1%}")


def main():
    skmap = load_skew()
    data = {s: prep_features(s) for s in ["QQQ", "SPY"]}

    print("=== DIRECTIONAL CONFIRMATION SPEED (keepers + candidates) ===")
    print(f"{'signal':<22s}{'n':>4s} {'P+1d':>6s} {'P+2d':>6s} {'P+3d':>6s} "
          f"{'MFE1%':>6s} {'1st+':>5s} {'hold':>5s}")
    conf_rows = []
    trade_store = {}

    for name, (sym, kw) in SIGNALS.items():
        df = data[sym]
        eq, tr = run_bt(df, **kw)
        if name == "IBS_G2_QQQ":
            rows = list(df.itertuples(index=False))
            idx = {d: i for i, d in enumerate(df["date"])}
            keep = []
            for t in tr.itertuples():
                i = idx.get(t.entry_date)
                if i and i > 0 and grade_v2(rows[i - 1], skmap) >= 2.5:
                    keep.append(t.Index)
            tr = tr.loc[keep].reset_index(drop=True)
        trade_store[name] = (df, tr)
        cs = confirmation_stats(df, tr)
        conf_rows.append((name, cs))
        print(f"{name:<22s}{cs['n']:>4d} {cs['p1']:>6.0%} {cs['p2']:>6.0%} {cs['p3']:>6.0%} "
              f"{cs['mfe1']:>6.0%} {cs['med_first']:>5.1f} {cs['med_hold']:>5.1f}")

    conf_rows.sort(key=lambda x: (-x[1]["p1"], -x[1]["mfe1"], x[1]["med_hold"]))
    print("\nRanked fastest confirmation: "
          + " > ".join(r[0] for r in conf_rows[:4]))

    print("\n=== SHORT PUT SPREAD SIMULATION (ATM / -2% OTM, BS model) ===")
    print(f"{'signal':<22s}{'n':>4s} {'WR':>6s} {'avgRisk':>8s} {'medRisk':>8s} "
          f"{'p25':>7s} {'p75':>7s} {'horizon':>7s}")
    for name in SIGNALS:
        df, tr = trade_store[name]
        if not len(tr):
            continue
        hor = max(1, int(round(tr["hold_days"].median())))
        res = put_spread_sim(df, tr, horizon=hor)
        if res:
            print(f"{name:<22s}{res['n']:>4d} {res['wr']:>6.0%} {res['avg']:>8.1%} "
                  f"{res['med']:>8.1%} {res['p25']:>7.1%} {res['p75']:>7.1%} {hor:>7d}d")

    bear_regime_check()
    tqqq_sleeve_check()


if __name__ == "__main__":
    main()
