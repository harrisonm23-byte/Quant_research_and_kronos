"""Build ranking, reliability flags, close-vs-open execution check, and report.md."""
import json
import os

import numpy as np
import pandas as pd

import run_suite                      # executes the suite on import
from engine import run_bt, compute_stats

OUT = os.path.dirname(os.path.abspath(__file__))
results = run_suite.results
bench = run_suite.bench
DATA = run_suite.DATA
RUNS = run_suite.RUNS

# --- same-close execution twins (lookahead version, for reliability flag) ---
close_cagr = {}
for run_id, strat, sym, kw in RUNS:
    kw2 = dict(kw)
    kw2["entry_fill"] = "close"
    kw2["exit_fill"] = "close"
    eq, trades = run_bt(DATA[sym], **kw2)
    close_cagr[run_id] = compute_stats(eq, trades, run_id)["cagr"]

# --- flags & ranking ---
AVAILABLE_SUBS = ["2010-2019", "2020-2021", "2022", "2023-present"]

rows = []
for st in results:
    rid = st["label"]
    sym = st["symbol"]
    subs = {k: v for k, v in st["subperiods"].items() if v is not None}
    sub_cagrs = {k: subs[k]["cagr"] for k in AVAILABLE_SUBS if k in subs}
    pos_subs = [k for k, v in sub_cagrs.items() if v > 0]
    stability = len(pos_subs) / len(sub_cagrs) if sub_cagrs else 0.0

    flags = []
    if st["n_trades"] < 50:
        flags.append("<50 trades")
    if st["total"] > 0 and len(pos_subs) == 1:
        flags.append("profits from one subperiod")
    if sub_cagrs.get("2023-present", 0) < 0 and st["total"] > 0:
        flags.append("collapsed post-2023")
    if st["maxdd"] < bench[sym]["maxdd"]:
        flags.append("maxDD worse than B&H")
    if st["pf"] < 1.2:
        flags.append("PF<1.2 after slippage")
    delta = close_cagr[rid] - st["cagr"]
    if delta > 0.02:
        flags.append(f"needs close-exec (+{delta:.1%} CAGR w/ lookahead)")

    rows.append(dict(run=rid, strategy=st["strategy"], symbol=sym,
                     cagr=st["cagr"], maxdd=st["maxdd"],
                     mar=st["cagr"] / abs(st["maxdd"]) if st["maxdd"] != 0 else 0.0,
                     sharpe=st["sharpe"], sortino=st["sortino"], wr=st["wr"], pf=st["pf"],
                     n=st["n_trades"], exposure=st["exposure"], avg_trade=st["avg_trade"],
                     med_trade=st["med_trade"], avg_hold=st["avg_hold"],
                     best=st["best"], worst=st["worst"],
                     stability=stability, close_delta=delta,
                     flags="; ".join(flags) if flags else "-",
                     n_flags=len(flags), sub_cagrs=sub_cagrs, yearly=st["yearly"]))

df = pd.DataFrame(rows)
# composite rank per the requested priority: PF, CAGR/maxDD, Sharpe, WR, #trades, stability
for col, asc in [("pf", False), ("mar", False), ("sharpe", False),
                 ("wr", False), ("n", False), ("stability", False)]:
    df[f"rk_{col}"] = df[col].rank(ascending=asc)
df["rank_score"] = df[[c for c in df.columns if c.startswith("rk_")]].mean(axis=1)
df = df.sort_values("rank_score").reset_index(drop=True)

# --- report.md ---
L = []
L.append("# Daily Strategy Backtest Suite — SPY / QQQ (Alpaca SIP, adjusted)")
L.append("")
L.append(f"- **Window:** 2017-04-01 → 2026-07-01 (~9.25y; Alpaca data starts 2016-01-04; "
         f"first 300 sessions reserved for indicator warmup)")
L.append("- **Execution:** signals on completed daily bars; fills next open; slippage 0.02%/side; commission $0; long-only; 100% equity/trade")
L.append("- **Unavailable subperiods:** 2000–2007, 2008–2009 (before Alpaca history). \"2010–2019\" = 2017–2019 here.")
L.append("- **Skipped:** TRIN Dip Buying (no NYSE TRIN data source available).")
L.append("")
L.append("## Benchmarks (buy & hold)")
L.append("")
L.append("| Symbol | CAGR | maxDD | Sharpe |")
L.append("|---|---|---|---|")
for sym, st in bench.items():
    L.append(f"| {sym} | {st['cagr']:.1%} | {st['maxdd']:.1%} | {st['sharpe']:.2f} |")
L.append("")
L.append("## Full ranking (composite of PF, CAGR/maxDD, Sharpe, WR, #trades, subperiod stability)")
L.append("")
L.append("| # | Run | CAGR | maxDD | MAR | Sharpe | Sortino | WR | PF | #tr | Expo | AvgTr | AvgHold | Stab | Flags |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, r in df.iterrows():
    L.append(f"| {i+1} | {r['run']} | {r['cagr']:.1%} | {r['maxdd']:.1%} | {r['mar']:.2f} | "
             f"{r['sharpe']:.2f} | {r['sortino']:.2f} | {r['wr']:.0%} | {r['pf']:.2f} | {r['n']} | "
             f"{r['exposure']:.0%} | {r['avg_trade']:.3%} | {r['avg_hold']:.1f}d | {r['stability']:.0%} | {r['flags']} |")
L.append("")
L.append("## Subperiod CAGR by run")
L.append("")
L.append("| Run | 2017-2019 | 2020-2021 | 2022 | 2023-present |")
L.append("|---|---|---|---|---|")
for _, r in df.iterrows():
    s = r["sub_cagrs"]
    L.append(f"| {r['run']} | " + " | ".join(
        f"{s.get(k, float('nan')):.1%}" if k in s else "n/a"
        for k in AVAILABLE_SUBS) + " |")
L.append("")
L.append("## Best/worst trades and yearly returns (top 10 by rank)")
L.append("")
for _, r in df.head(10).iterrows():
    yr = ", ".join(f"{k}: {v:+.1%}" for k, v in sorted(r["yearly"].items()))
    L.append(f"- **{r['run']}** — best {r['best']:+.1%}, worst {r['worst']:+.1%}, median trade "
             f"{r['med_trade']:+.3%}. Yearly: {yr}")
L.append("")
L.append("## Notes")
L.append("")
L.append("- Trade logs: `trades_<run>.csv`; equity curves: `equity_<run>.csv` (same directory).")
L.append("- `Stab` = fraction of available subperiods with positive CAGR.")
L.append("- The close-exec flag compares against a same-close fill variant (which has lookahead); "
         "strategies carrying it depend on untradeable fills for a chunk of their edge.")
L.append("- Triple RSI reliability: see run S7 — Wilson 95% CI reported in console output.")

with open(os.path.join(OUT, "report.md"), "w") as f:
    f.write("\n".join(L))

df.drop(columns=["sub_cagrs", "yearly"]).to_csv(os.path.join(OUT, "ranking.csv"), index=False)

# Triple RSI reliability detail
s7 = df[df["run"] == "S7_TripleRSI_SPY"].iloc[0]
n7, wr7 = int(s7["n"]), s7["wr"]
z = 1.96
ph = wr7
den = 1 + z * z / n7
ctr = ph + z * z / (2 * n7)
mg = z * np.sqrt(ph * (1 - ph) / n7 + z * z / (4 * n7 * n7))
print(f"TripleRSI: {n7} trades over 9.25y -> {9.25*365.25/n7:.0f} days between trades; "
      f"WR={wr7:.1%}, Wilson 95% CI [{(ctr-mg)/den:.1%}, {(ctr+mg)/den:.1%}]")
print("\nTop 12 by composite rank:")
print(df.head(12)[["run", "cagr", "maxdd", "sharpe", "wr", "pf", "n", "stability", "flags"]].to_string(index=False))
print("\nreport.md written")
