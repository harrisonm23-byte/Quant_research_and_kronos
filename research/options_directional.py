"""Directional options overlay: cheap OTM calls + early spike exit.

Contrasts with put-spread / hold-to-horizon approach. Structures:
  A. OTM call only — buy 2% OTM weekly, sell on first intraday touch of
     +target% within max_sessions, else sell at close of last session.
  B. OTM call ladder — 2 contracts at 2% OTM (2x premium at risk).
  C. Hedged — 2x 2% OTM calls + 1x ATM put (protective); exit all legs
     together on spike or time stop.

Pricing: Black-Scholes, IV = rv20 * iv_mult, 2% leg haircut, iv beta on exit.
"""
import math
import os

import numpy as np
import pandas as pd

from engine import load_symbol, run_bt

OUT = os.path.dirname(os.path.abspath(__file__))
R = 0.04
COST = 0.02


def _nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


def _N(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, iv, kind):
    if T <= 0:
        return max(S - K, 0.0) if kind == "c" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (R + iv * iv / 2) * T) / (iv * math.sqrt(T))
    d2 = d1 - iv * math.sqrt(T)
    if kind == "c":
        return S * _N(d1) - K * math.exp(-R * T) * _N(d2)
    return K * math.exp(-R * T) * _N(-d2) - S * _N(-d1)


def iv_exit(iv0, ret):
    return iv0 * min(1.4, max(0.6, 1 - 3 * ret))


SIGNAL_FNS = {
    "5DayLow_A": dict(
        entry_fn=lambda r: _nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5,
        exit_fn=lambda r: _nn(r.prev_close) and r.close > r.prev_close),
    "IBS": dict(
        entry_fn=lambda r: r.ibs < 0.20,
        exit_fn=lambda r: r.ibs > 0.70),
    "DoubleSeven": dict(
        entry_fn=lambda r: _nn(r.sma200, r.lc7) and r.close > r.sma200 and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7),
    "TT_A": dict(
        entry_fn=lambda r: r.weekday == 0 and r.close < r.open,
        exit_fn=None, max_hold=1),
}


def prep(sym):
    df = load_symbol(sym)
    c = df["close"]
    df["lc5"] = c.rolling(5).min()
    df["lc7"] = c.rolling(7).min()
    df["hc7"] = c.rolling(7).max()
    df["sma200"] = c.rolling(200).mean()
    df["prev_close"] = c.shift(1)
    rng = df["high"] - df["low"]
    df["ibs"] = np.where(rng > 0, (c - df["low"]) / rng, 0.5)
    df["rv20"] = np.log(c / c.shift(1)).rolling(20).std() * math.sqrt(252)
    return df


def first_spike_bar(h, i, max_sessions, S0, target_pct):
    for k in range(i, min(i + max_sessions, len(h))):
        if h[k] >= S0 * (1 + target_pct):
            return k
    return None


def simulate_trade(S0, o, h, c, rv, i, expiry_sessions, otm_pct, target_pct,
                   max_sessions, structure, iv_mult=1.0):
    """
    structure: 'single' | 'ladder' | 'hedged'
    Returns P&L as fraction of net debit paid (or per-call for reporting).
    """
    if math.isnan(rv):
        return None
    iv0 = rv * iv_mult
    K_call = S0 * (1 + otm_pct)
    K_put = S0
    T0 = expiry_sessions / 252.0
    j_end = min(i + max_sessions - 1, len(c) - 1)

    spike_k = first_spike_bar(h, i, max_sessions, S0, target_pct)

    if spike_k is not None:
        elapsed = (spike_k - i) + 0.5
        Sx = S0 * (1 + target_pct)
        ret = target_pct
    else:
        elapsed = max_sessions
        Sx = c[j_end]
        ret = Sx / S0 - 1
        spike_k = j_end

    Tx = max(expiry_sessions - elapsed, 0.05) / 252.0
    ivx = iv_exit(iv0, ret)

    # entry premiums
    c_otm = bs(S0, K_call, T0, iv0, "c")
    p_atm = bs(S0, K_put, T0, iv0, "p")

    if structure == "single":
        debit = c_otm * (1 + COST / 2)
        cx = bs(Sx, K_call, Tx, ivx, "c")
        pnl = (cx * (1 - COST / 2) - debit) / debit if debit > 0 else None
    elif structure == "ladder":
        n = 2
        debit = n * c_otm * (1 + COST / 2)
        cx = n * bs(Sx, K_call, Tx, ivx, "c")
        pnl = (cx * (1 - COST / 2) - debit) / debit if debit > 0 else None
    elif structure == "hedged":
        n = 2
        debit = n * c_otm * (1 + COST / 2) + p_atm * (1 + COST / 2)
        cx = n * bs(Sx, K_call, Tx, ivx, "c")
        px = bs(Sx, K_put, Tx, ivx, "p")
        credit = (cx + px) * (1 - COST / 2)
        pnl = (credit - debit) / debit if debit > 0 else None
    elif structure == "risk_rev":
        # buy OTM call, sell ATM put (financed directional)
        credit_put = p_atm * (1 - COST / 2)
        debit_call = c_otm * (1 + COST / 2)
        net = debit_call - credit_put  # often small debit or credit
        cx = bs(Sx, K_call, Tx, ivx, "c")
        px = bs(Sx, K_put, Tx, ivx, "p")
        value = cx * (1 - COST / 2) - px * (1 + COST / 2)  # long call short put
        entry_cost = max(debit_call, 0.01 * S0)  # floor denom
        pnl = (value - (debit_call - credit_put)) / entry_cost
    else:
        return None
    hit_target = spike_k is not None and spike_k < i + max_sessions
    return pnl, hit_target


def run_grid(df, trades, label):
    o, h, c = df["open"].values, df["high"].values, df["close"].values
    rv = df["rv20"].values
    idx = {d: i for i, d in enumerate(df["date"])}

    configs = [
        ("single 2%OTM +0.75% 2d", "single", 0.02, 0.0075, 2, 5),
        ("single 2%OTM +1.0% 2d", "single", 0.02, 0.010, 2, 5),
        ("single 2%OTM +1.0% 1d", "single", 0.02, 0.010, 1, 4),
        ("ladder 2x2%OTM +0.75% 2d", "ladder", 0.02, 0.0075, 2, 5),
        ("hedged 2c+1p +0.75% 2d", "hedged", 0.02, 0.0075, 2, 5),
        ("hedged 2c+1p +1.0% 2d", "hedged", 0.02, 0.010, 2, 5),
    ]
    print(f"\n--- {label} (n={len(trades)}) ---")
    print(f"{'config':<28s}{'n':>4s}{'WR':>6s}{'avg':>8s}{'med':>8s}{'hit%':>6s}{'p75':>8s}")
    for cfg_name, struct, otm, tgt, msess, exp in configs:
        pnls, hits = [], []
        for t in trades.itertuples():
            i = idx.get(t.entry_date)
            if i is None:
                continue
            S0 = o[i]
            out = simulate_trade(S0, o, h, c, rv[i], i, exp, otm, tgt, msess, struct)
            if out is None:
                continue
            pnl, hit = out
            pnls.append(pnl)
            hits.append(hit)
        if not pnls:
            continue
        a = np.array(pnls)
        print(f"{cfg_name:<28s}{len(a):>4d}{(a>0).mean():>6.0%}{a.mean():>8.0%}"
              f"{np.median(a):>8.0%}{np.mean(hits):>6.0%}{np.percentile(a,75):>8.0%}")


def main():
    df = prep("QQQ")
    print("=== DIRECTIONAL OTM OPTIONS (early spike exit, QQQ signals) ===")
    print("P&L = return on net debit. hit% = fraction exiting on target touch (not time stop).")
    print("Compare to hold-to-horizon put spreads in options_overlay_suite.py\n")

    for name, kw in SIGNAL_FNS.items():
        _, tr = run_bt(df, **kw)
        run_grid(df, tr, name)

    # MFE context: how often is +0.75% / +1% reachable in 1-2 sessions?
    print("\n=== UNDERLYING SPIKE REACHABILITY (informs exit targets) ===")
    idx = {d: i for i, d in enumerate(df["date"])}
    o, h = df["open"].values, df["high"].values
    for name, kw in SIGNAL_FNS.items():
        _, tr = run_bt(df, **kw)
        r75, r100 = [], []
        for t in tr.itertuples():
            i = idx.get(t.entry_date)
            if i is None:
                continue
            S0 = o[i]
            w = h[i:min(i + 2, len(h))]
            r75.append((w.max() / S0 - 1) >= 0.0075)
            r100.append((w.max() / S0 - 1) >= 0.010)
        print(f"  {name:<14s} P(+0.75% in 2d)={np.mean(r75):.0%}  P(+1.0% in 2d)={np.mean(r100):.0%}")


if __name__ == "__main__":
    main()
