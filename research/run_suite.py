"""Run the full 10-strategy daily backtest suite on SPY/QQQ Alpaca data."""
import json
import math
import os

import numpy as np
import pandas as pd

import engine
from engine import run_bt, compute_stats, subperiod_stats, load_symbol, STAT_START

OUT = os.path.dirname(os.path.abspath(__file__))
DATA = {s: load_symbol(s) for s in ["SPY", "QQQ"]}


def nn(*vals):
    return all(not (isinstance(v, float) and math.isnan(v)) for v in vals)


RUNS = []   # (run_id, strategy, symbol, df, kwargs)

# --- 1. Double Seven (trend-filtered daily version) ---
for sym in ["SPY", "QQQ"]:
    RUNS.append((f"S1_DoubleSeven_{sym}", "1 Double Seven", sym, dict(
        entry_fn=lambda r: nn(r.sma200, r.lc7) and r.close > r.sma200 and r.close <= r.lc7,
        exit_fn=lambda r: r.close >= r.hc7)))

# --- 2. Connors RSI2 Original: CumRSI2 < 5, exit close > SMA5 ---
for sym in ["SPY", "QQQ"]:
    RUNS.append((f"S2_RSI2Orig_{sym}", "2 RSI2 Original", sym, dict(
        entry_fn=lambda r: nn(r.sma200, r.cumrsi2) and r.close > r.sma200 and r.cumrsi2 < 5,
        exit_fn=lambda r: r.close > r.sma5)))

# --- 3. Connors RSI2 Modified: CumRSI2 < 10, exit close > SMA10; A no stop, B -2% stop ---
for sym in ["SPY", "QQQ"]:
    for tag, stop in [("A_nostop", None), ("B_stop2", 0.02)]:
        RUNS.append((f"S3_RSI2Mod_{sym}_{tag}", "3 RSI2 Modified", sym, dict(
            entry_fn=lambda r: nn(r.sma200, r.cumrsi2) and r.close > r.sma200 and r.cumrsi2 < 10,
            exit_fn=lambda r: r.close > r.sma10, stop_pct=stop)))

# --- 4. IBS Basic: entry/exit threshold grid, both symbols ---
for sym in ["QQQ", "SPY"]:
    for lo in [0.20, 0.25, 0.30]:
        for hi in [0.70, 0.75, 0.80]:
            RUNS.append((f"S4_IBS_{sym}_e{int(lo*100)}_x{int(hi*100)}", "4 IBS Basic", sym, dict(
                entry_fn=(lambda lo_: lambda r: r.ibs < lo_)(lo),
                exit_fn=(lambda hi_: lambda r: r.ibs > hi_)(hi))))

# --- 5. QQQ Lower Band + IBS; A no regime, B SMA300 regime filter ---
RUNS.append(("S5_LowerBand_QQQ_A", "5 Lower Band IBS", "QQQ", dict(
    entry_fn=lambda r: nn(r.lower_band) and r.close < r.lower_band and r.ibs < 0.30,
    exit_fn=lambda r: nn(r.prev_high) and r.close > r.prev_high)))
RUNS.append(("S5_LowerBand_QQQ_B_sma300", "5 Lower Band IBS", "QQQ", dict(
    entry_fn=lambda r: nn(r.lower_band) and r.close < r.lower_band and r.ibs < 0.30,
    exit_fn=lambda r: nn(r.prev_high) and r.close > r.prev_high,
    regime_fn=lambda r: nn(r.sma300) and r.close < r.sma300)))

# --- 6. Turnaround Tuesday ---
for sym in ["SPY", "QQQ"]:
    RUNS.append((f"S6_TT_A_{sym}", "6 Turnaround Tuesday", sym, dict(
        entry_fn=lambda r: r.weekday == 0 and r.close < r.open,
        exit_fn=None, max_hold=1)))                       # exit Wednesday open
    RUNS.append((f"S6_TT_Atheo_{sym}", "6 Turnaround Tuesday", sym, dict(
        entry_fn=lambda r: r.weekday == 0 and r.close < r.open,
        exit_fn=None, max_hold=1, exit_fill="close")))    # theoretical Tuesday close
    RUNS.append((f"S6_TT_B_{sym}", "6 Turnaround Tuesday", sym, dict(
        entry_fn=lambda r: r.weekday == 0 and nn(r.prev_low) and r.close < r.prev_low,
        exit_fn=None, max_hold=1)))
    RUNS.append((f"S6_TT_C_{sym}", "6 Turnaround Tuesday", sym, dict(
        entry_fn=lambda r: (r.weekday == 0 and nn(r.prev_close, r.prev2_close)
                            and r.close < r.prev_close and r.prev_close < r.prev2_close),
        exit_fn=lambda r: nn(r.prev_high) and r.close > r.prev_high, max_hold=5)))

# --- 7. Triple RSI (SPY only) ---
RUNS.append(("S7_TripleRSI_SPY", "7 Triple RSI", "SPY", dict(
    entry_fn=lambda r: (nn(r.sma200, r.rsi5, r.rsi5_1, r.rsi5_2, r.rsi5_3)
                        and r.close > r.sma200 and r.rsi5 < 30
                        and r.rsi5 < r.rsi5_1 and r.rsi5_1 < r.rsi5_2
                        and r.rsi5_3 < 60),
    exit_fn=lambda r: r.rsi5 > 50)))

# --- 8. SPY IBS + RSI21 Classical ---
RUNS.append(("S8_IBSRSI21_SPY", "8 IBS+RSI21 Classical", "SPY", dict(
    entry_fn=lambda r: nn(r.rsi21) and r.ibs < 0.25 and r.rsi21 < 45,
    exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)))

# --- 9. Five-Day Low + Low IBS ---
for sym in ["SPY", "QQQ"]:
    RUNS.append((f"S9_5DayLow_A_{sym}", "9 Five-Day Low", sym, dict(
        entry_fn=lambda r: nn(r.lc5) and r.ibs < 0.25 and r.close <= r.lc5,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)))
    RUNS.append((f"S9_5DayLow_B_{sym}", "9 Five-Day Low", sym, dict(
        entry_fn=lambda r: nn(r.ll5) and r.ibs < 0.25 and r.low <= r.ll5,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)))

# --- 10. Large Down-Day Bounce ---
for sym in ["SPY", "QQQ"]:
    RUNS.append((f"S10_DownDay_A_{sym}", "10 Down-Day Bounce", sym, dict(
        entry_fn=lambda r: nn(r.ret1) and r.ret1 <= -0.05, exit_fn=None, max_hold=1)))
    RUNS.append((f"S10_DownDay_B_{sym}", "10 Down-Day Bounce", sym, dict(
        entry_fn=lambda r: nn(r.ret1) and r.ret1 <= -0.05, exit_fn=None, max_hold=2)))
    RUNS.append((f"S10_DownDay_C_{sym}", "10 Down-Day Bounce", sym, dict(
        entry_fn=lambda r: nn(r.ret1) and r.ret1 <= -0.05,
        exit_fn=lambda r: nn(r.prev_close) and r.close > r.prev_close)))

# TRIN strategy: SKIPPED — no NYSE TRIN/Arms Index data source available via Alpaca.

# ---------------- execute ----------------
results = []
for run_id, strat, sym, kw in RUNS:
    df = DATA[sym]
    eq, trades = run_bt(df, **kw)
    st = compute_stats(eq, trades, run_id)
    st["strategy"] = strat
    st["symbol"] = sym
    st["subperiods"] = {k: (v if v is None else {kk: vv for kk, vv in v.items() if kk != "yearly"})
                        for k, v in subperiod_stats(eq, trades).items()}
    results.append(st)
    # persist artifacts
    if len(trades):
        trades.to_csv(os.path.join(OUT, f"trades_{run_id}.csv"), index=False)
    eq.to_csv(os.path.join(OUT, f"equity_{run_id}.csv"))

# benchmarks
bench = {}
for sym in ["SPY", "QQQ"]:
    df = DATA[sym]
    eq, trades = run_bt(df, entry_fn=lambda r: True, exit_fn=lambda r: False, slip=0.0)
    st = compute_stats(eq, trades if len(trades) else pd.DataFrame(columns=["ret", "hold_days", "exit_date"]),
                       f"BH_{sym}")
    st["subperiods"] = {k: (v if v is None else {kk: vv for kk, vv in v.items() if kk != "yearly"})
                        for k, v in subperiod_stats(eq, pd.DataFrame(columns=["ret", "hold_days", "exit_date"])).items()}
    bench[sym] = st

with open(os.path.join(OUT, "suite_results.json"), "w") as f:
    json.dump({"results": results, "benchmarks": bench,
               "meta": {"window": f"{STAT_START.date()} -> 2026-07-01",
                        "slippage_per_side": engine.SLIP,
                        "data": "Alpaca SIP daily, adjustment=all",
                        "skipped": ["TRIN Dip Buying (no TRIN data source)"],
                        "unavailable_subperiods": ["2000-2007", "2008-2009 (Alpaca data starts 2016)"]}},
              f, indent=2, default=str)

# console summary
print(f"{'run':<28s} {'CAGR':>7s} {'maxDD':>7s} {'Sharpe':>6s} {'WR':>6s} {'PF':>5s} {'#tr':>5s} {'expo':>6s} {'avgtr':>7s}")
for st in results:
    print(f"{st['label']:<28s} {st['cagr']:>7.1%} {st['maxdd']:>7.1%} {st['sharpe']:>6.2f} "
          f"{st['wr']:>6.1%} {st['pf']:>5.2f} {st['n_trades']:>5d} {st['exposure']:>6.1%} {st['avg_trade']:>7.3%}")
print()
for sym, st in bench.items():
    print(f"{'BH_'+sym:<28s} {st['cagr']:>7.1%} {st['maxdd']:>7.1%} {st['sharpe']:>6.2f}")
