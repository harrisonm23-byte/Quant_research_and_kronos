"""Simulate options structures on TT-A/QQQ and FiveDayLow-A/QQQ signal dates.

Structures per signal:
  A. Long ATM call, weekly expiry (T=4-5 trading days), exit at the strategy's
     fixed horizon (TT-A: Wednesday close; 5DL: day-3 close).
  A2. Same, but "spike exit": if intraday high reaches +1% over entry within
     the horizon, exit there (mid-session); else exit at horizon close.
  B. Short put spread: short ATM put / long 2% OTM put, same expiry, exit at
     the same horizon.

Pricing: Black-Scholes, r=4%, IV = 20d realized vol * iv_mult, with a
spot-vol beta at exit: IV_exit = IV_entry * clamp(1 - 3*underlying_ret, .6, 1.4).
Costs: 2% of premium round-trip haircut on every leg.
No real options data — treat results as structural comparison, not P&L forecast.
"""
import math
import os

import numpy as np
import pandas as pd

from engine import load_symbol

OUT = os.path.dirname(os.path.abspath(__file__))
R = 0.04
COST = 0.02          # round-trip, fraction of each leg's premium


def N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, iv, kind):
    if T <= 0:
        intr = max(S - K, 0.0) if kind == "c" else max(K - S, 0.0)
        return intr
    d1 = (math.log(S / K) + (R + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "c":
        return S * N(d1) - K * math.exp(-R * T) * N(d2)
    return K * math.exp(-R * T) * N(-d2) - S * N(-d1)


def iv_exit(iv0, ret):
    return iv0 * min(1.4, max(0.6, 1 - 3 * ret))


df = load_symbol("QQQ")
df["rv20"] = np.log(df["close"] / df["close"].shift(1)).rolling(20).std() * math.sqrt(252)
idx = {d: i for i, d in enumerate(df["date"])}
o = df["open"].values
h = df["high"].values
c = df["close"].values
rv = df["rv20"].values
nb = len(df)

SIGNALS = {
    # (trades csv, horizon sessions incl entry day, expiry sessions from entry)
    "TT_A_QQQ": ("trades_S6_TT_A_QQQ.csv", 2, 4),
    "5DayLow_A_QQQ": ("trades_S9_5DayLow_A_QQQ.csv", 3, 5),
}


def simulate(sig, iv_mult):
    path, hor, expiry = SIGNALS[sig]
    tr = pd.read_csv(os.path.join(OUT, path), parse_dates=["entry_date"])
    res = {"call": [], "call_spike": [], "putspread": []}
    for t in tr.itertuples():
        i = idx.get(t.entry_date)
        if i is None or i + hor - 1 >= nb or math.isnan(rv[i]):
            continue
        S0 = o[i]
        iv0 = rv[i] * iv_mult
        K = S0
        K2 = 0.98 * S0
        T0 = expiry / 252.0
        j = i + hor - 1
        S1 = c[j]
        ret = S1 / S0 - 1
        T1 = max(expiry - hor, 0) / 252.0
        ive = iv_exit(iv0, ret)

        # A. long ATM call, horizon exit
        c0 = bs(S0, K, T0, iv0, "c")
        c1 = bs(S1, K, T1, ive, "c")
        pnl = (c1 * (1 - COST / 2) - c0 * (1 + COST / 2)) / c0
        res["call"].append(pnl)

        # A2. spike exit at +1% if touched
        spike_i = None
        for kdx in range(i, j + 1):
            if h[kdx] >= S0 * 1.01:
                spike_i = kdx
                break
        if spike_i is not None:
            elapsed = (spike_i - i) + 0.5
            Ts = max(expiry - elapsed, 0.1) / 252.0
            Ss = S0 * 1.01
            cs = bs(Ss, K, Ts, iv_exit(iv0, 0.01), "c")
            pnl2 = (cs * (1 - COST / 2) - c0 * (1 + COST / 2)) / c0
        else:
            pnl2 = pnl
        res["call_spike"].append(pnl2)

        # B. short put spread (short ATM, long 2% OTM)
        p0s = bs(S0, K, T0, iv0, "p")
        p0l = bs(S0, K2, T0, iv0, "p")
        credit = (p0s - p0l) * (1 - COST / 2)
        p1s = bs(S1, K, T1, ive, "p")
        p1l = bs(S1, K2, T1, ive, "p")
        cost_to_close = (p1s - p1l) * (1 + COST / 2)
        max_risk = (K - K2) - credit
        res["putspread"].append((credit - cost_to_close) / max_risk)
    return res


print(f"{'signal':<16s} {'structure':<12s} {'ivx':>4s} {'n':>4s} {'WR':>6s} "
      f"{'avg':>8s} {'med':>8s} {'p25':>8s} {'p75':>8s} {'sum':>8s}")
for sig in SIGNALS:
    for iv_mult in [1.0, 1.2, 1.4]:
        res = simulate(sig, iv_mult)
        for struct, pnls in res.items():
            a = np.array(pnls)
            print(f"{sig:<16s} {struct:<12s} {iv_mult:>4.1f} {len(a):>4d} {(a>0).mean():>6.0%} "
                  f"{a.mean():>8.1%} {np.median(a):>8.1%} {np.percentile(a,25):>8.1%} "
                  f"{np.percentile(a,75):>8.1%} {a.sum():>8.0%}")
    print()
print("call/call_spike: P&L as % of premium paid.  putspread: P&L as % of max risk.")
print("ivx = IV / 20d realized vol multiplier. sum = cumulative % over ~9.25y of signals.")
