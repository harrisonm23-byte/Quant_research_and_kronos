"""Cross-asset backtest for the three equity keeper strategies on SPY, QQQ, TQQQ.

Also prints fixed-horizon stats useful for options overlay selection and a short
put-spread simulation on keeper signals (Black-Scholes, same model as options_sim.py).

Usage:
  python3 keeper_cross_asset.py
  python3 keeper_cross_asset.py --data-dir /path/to/csvs

Expects {SYM}_daily.csv with columns: date, open, high, low, close, volume.
If missing, fetches via examples/fetch_market_data.py into research/.
"""
import argparse
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from engine import STAT_START, compute_stats, run_bt

OUT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(OUT)

KEEPERS = [
    ("IBS<.20/.70", dict(
        entry_fn=lambda r: r.ibs < 0.20,
        exit_fn=lambda r: r.ibs > 0.70)),
    ("DoubleSeven", dict(
        entry_fn=lambda r: _nn(r.sma200, r.lc7) and r.close > r.sma200 and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7)),
    ("5DayLow-A", dict(
        entry_fn=lambda r: _nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5,
        exit_fn=lambda r: _nn(r.prev_close) and r.close > r.prev_close)),
]

SYMBOLS = ["SPY", "QQQ", "TQQQ"]
HALVES = [("2017-2021", "2017-04-01", "2022-01-01"), ("2022-2026", "2022-01-01", "2027-01-01")]

# Black-Scholes helpers (options_sim.py)
R = 0.04
COST = 0.02


def _nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


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


def ensure_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    missing = [s for s in SYMBOLS if not os.path.exists(os.path.join(data_dir, f"{s}_daily.csv"))]
    if not missing:
        return
    print(f"Fetching missing daily data: {', '.join(missing)}")
    tmp = os.path.join(data_dir, "_fetch")
    os.makedirs(tmp, exist_ok=True)
    cmd = [sys.executable, os.path.join(REPO, "examples", "fetch_market_data.py"),
           *missing, "--range", "10Y", "--outdir", tmp]
    subprocess.check_call(cmd)
    for sym in missing:
        src = os.path.join(tmp, f"{sym}.csv")
        dst = os.path.join(data_dir, f"{sym}_daily.csv")
        df = pd.read_csv(src)
        df = df.rename(columns={"timestamps": "date"})
        df[["date", "open", "high", "low", "close", "volume"]].to_csv(dst, index=False)
        print(f"  wrote {dst}")


def load_symbol_from(data_dir, sym):
    df = pd.read_csv(os.path.join(data_dir, f"{sym}_daily.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["sma200"] = c.rolling(200).mean()
    df["lc7"] = c.rolling(7).min()
    df["hc7"] = c.rolling(7).max()
    df["lc5"] = c.rolling(5).min()
    rng = h - l
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    df["prev_close"] = c.shift(1)
    return df


def period_stats(eq, trades, start, end):
  sl = eq[(eq.index >= pd.Timestamp(start)) & (eq.index < pd.Timestamp(end))]
  if len(sl) < 20:
    return None
  tr = trades[(trades["exit_date"] >= start) & (trades["exit_date"] < end)] if len(trades) else trades
  return compute_stats(sl, tr)


def run_all(data_dir):
    data = {s: load_symbol_from(data_dir, s) for s in SYMBOLS}
    rows = []
    trades_store = {}

    for strat, kw in KEEPERS:
        for sym in SYMBOLS:
            eq, tr = run_bt(data[sym], **kw)
            st = compute_stats(eq, tr, f"{strat}_{sym}")
            trades_store[(strat, sym)] = (data[sym], tr)
            half1 = half2 = (float("nan"), float("nan"), float("nan"))
            for half, lo, hi in HALVES:
                ps = period_stats(eq, tr, lo, hi)
                tup = (ps["cagr"], ps["wr"], ps["pf"]) if ps else (float("nan"), float("nan"), float("nan"))
                if half == HALVES[0][0]:
                    half1 = tup
                else:
                    half2 = tup
            rows.append([strat, sym, st["cagr"], st["wr"], st["pf"], st["maxdd"],
                         st["sharpe"], st["avg_trade"], st["med_trade"],
                         st["avg_hold"], st["n_trades"], st["exposure"],
                         *half1[:1], *half2[:1]])

    # buy & hold
    for sym in SYMBOLS:
        d = data[sym][data[sym]["date"] >= STAT_START]
        eq = pd.Series(d["close"].values, index=pd.DatetimeIndex(d["date"]))
        days = (eq.index[-1] - eq.index[0]).days
        ann = (eq.iloc[-1] / eq.iloc[0]) ** (365.25 / days) - 1
        peak = eq.cummax()
        maxdd = ((eq - peak) / peak).min()
        dr = eq.pct_change().dropna()
        sharpe = dr.mean() / dr.std() * math.sqrt(252) if dr.std() > 0 else 0.0
        rows.append(["Buy&Hold", sym, ann, float("nan"), float("nan"), maxdd, sharpe,
                     float("nan"), float("nan"), float("nan"), float("nan"), float("nan"),
                     float("nan"), float("nan"), float("nan")])

    hdr = (f"{'Strategy':<14s}{'Sym':<6s}{'Ann%':>7s}{'WR%':>6s}{'PF':>6s}{'MaxDD':>7s}"
           f"{'Sharpe':>7s}{'AvgTr':>7s}{'MedTr':>7s}{'Hold':>6s}{'#Tr':>5s}{'Expo':>6s}"
           f"{'H1Ann':>7s}{'H2Ann':>7s}")
    print("=== KEEPER STRATEGIES: SPY vs QQQ vs TQQQ ===")
    print(f"Window: {STAT_START.date()} -> latest | fill next open | slippage 0.02%/side")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        def f(x, w, dec=1, pct=True):
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return f"{'--':>{w}s}"
            return f"{x*100:>{w}.{dec}f}" if pct else f"{x:>{w}.{dec}f}"
        ntr = "--" if r[10] is None or (isinstance(r[10], float) and math.isnan(r[10])) else f"{int(r[10]):>5d}"
        print(f"{r[0]:<14s}{r[1]:<6s}{f(r[2],7)}{f(r[3],6)}{f(r[4],6,2)}{f(r[5],7)}"
              f"{f(r[6],7,2,False)}{f(r[7],7,3)}{f(r[8],7,3)}{f(r[9],6,1,False)}"
              f"{ntr}{f(r[11],6)}{f(r[12],7)}{f(r[13],7)}")

    # scaled TQQQ: 1/3 notional per trade (portfolio return = 1 + trade_ret/3 per round-trip)
    print("\n=== TQQQ at 1/3 NOTIONAL (1 + ret/3 per completed trade) ===")
    print(f"{'Strategy':<14s}{'Ann%':>7s}{'MaxDD':>7s}{'WR%':>6s}{'PF':>6s}{'#Tr':>5s}")
    for strat, _ in KEEPERS:
        _, tr = trades_store[(strat, "TQQQ")]
        if not len(tr):
            continue
        port_rets = tr["ret"].values / 3.0
        eq_curve = np.cumprod(1 + port_rets)
        days = (data["TQQQ"]["date"].iloc[-1] - pd.Timestamp(STAT_START)).days
        ann = eq_curve[-1] ** (365.25 / days) - 1
        wr = (port_rets > 0).mean()
        wins, losses = port_rets[port_rets > 0], port_rets[port_rets <= 0]
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        peak = np.maximum.accumulate(eq_curve)
        maxdd = (eq_curve / peak - 1).min()
        print(f"{strat:<14s}{ann*100:>7.1f}{maxdd*100:>7.1f}{wr*100:>6.1f}{pf:>6.2f}{len(tr):>5d}")

    # options lens
    print("\n=== OPTIONS LENS (fixed-horizon from next-open entry) ===")
    hdr2 = f"{'strat+sym':<22s}{'n':>4s} " + " ".join(f"{'P+'+str(h)+'d':>6s}" for h in [1,2,3,5])
    hdr2 += f" {'med3d':>7s} {'avg3d':>7s} {'MFE1%':>7s} {'MFE2%':>7s} {'hold':>5s}"
    print(hdr2)
    for strat, _ in KEEPERS:
        for sym in SYMBOLS:
            df, tr = trades_store[(strat, sym)]
            if not len(tr):
                continue
            idx = {d: i for i, d in enumerate(df["date"])}
            closes, highs = df["close"].values, df["high"].values
            nb = len(df)
            horizons = {h: [] for h in [1, 2, 3, 5]}
            mfe3, holds = [], []
            for t in tr.itertuples():
                i = idx.get(t.entry_date)
                if i is None:
                    continue
                holds.append(t.hold_days)
                for h in [1, 2, 3, 5]:
                    j = i + h - 1
                    if j < nb:
                        horizons[h].append(closes[j] / t.entry_px - 1)
                j3 = min(i + 2, nb - 1)
                mfe3.append(highs[i:j3 + 1].max() / t.entry_px - 1)
            r3 = np.array(horizons[3])
            mfe = np.array(mfe3)
            label = f"{strat} {sym}"
            line = f"{label:<22s}{len(mfe):>4d} "
            line += " ".join(f"{(np.array(horizons[h])>0).mean():>6.0%}" for h in [1,2,3,5])
            line += f" {np.median(r3):>+7.2%} {r3.mean():>+7.2%}"
            line += f" {(mfe>=0.01).mean():>7.0%} {(mfe>=0.02).mean():>7.0%}"
            line += f" {np.median(holds):>5.1f}"
            print(line)

    # put spread sim on keepers
    print("\n=== SHORT PUT SPREAD SIM (ATM / -2% OTM, horizon = median hold, iv=1.0x RV20) ===")
    print(f"{'strat+sym':<22s}{'n':>4s}{'WR':>6s}{'avg%risk':>9s}{'med%risk':>9s}{'sum':>8s}")
    for strat, _ in KEEPERS:
        for sym in ["QQQ", "SPY"]:
            df, tr = trades_store[(strat, sym)]
            if not len(tr):
                continue
            rv = np.log(df["close"] / df["close"].shift(1)).rolling(20).std() * math.sqrt(252)
            idx = {d: i for i, d in enumerate(df["date"])}
            o, h, c = df["open"].values, df["high"].values, df["close"].values
            nb = len(df)
            med_hold = max(1, int(round(tr["hold_days"].median())))
            expiry = max(med_hold + 2, 5)
            pnls = []
            for t in tr.itertuples():
                i = idx.get(t.entry_date)
                if i is None or math.isnan(rv.iloc[i]):
                    continue
                j = min(i + med_hold - 1, nb - 1)
                S0, S1 = o[i], c[j]
                iv0 = rv.iloc[i]
                K, K2 = S0, 0.98 * S0
                T0, T1 = expiry / 252.0, max(expiry - med_hold, 0) / 252.0
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
            if pnls:
                a = np.array(pnls)
                label = f"{strat} {sym}"
                print(f"{label:<22s}{len(a):>4d}{(a>0).mean():>6.0%}{a.mean():>9.1%}"
                      f"{np.median(a):>9.1%}{a.sum():>8.0%}")


def main():
    p = argparse.ArgumentParser(description="Cross-asset keeper backtest + options lens")
    p.add_argument("--data-dir", default=OUT, help="Directory with {SYM}_daily.csv files")
    args = p.parse_args()
    ensure_data(args.data_dir)
    run_all(args.data_dir)


if __name__ == "__main__":
    main()
