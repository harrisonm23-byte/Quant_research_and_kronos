"""Stage 3: Warrior patterns on the small-cap gapper universe (native habitat).

Per event day: 1m bars 04:00-20:00 ET. Premarket high = 04:00-09:29 high.
Detectors (on RTH bars): gap-and-go premarket-high breakout (1m), flat-top breakout,
failed flat-top, 5-candle 5m reversal, 10-candle 1m reversal — user's detector code.
Execution: next-bar-open, costs 30bps round trip (small-cap spreads). R-scaled races:
+/-2% barriers, 60-minute window. Buckets: gap size, price, rel-vol, year.
CAVEAT: survivorship (current listings only) — flatters longs, dampens shorts.
"""
import os, sys
from datetime import time as dtime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "warrior_backtest"))
from warrior_pattern_backtest import (consecutive_candle_reversals, flat_top_breakouts,
                                      failed_flat_top_breakouts)

NY = ZoneInfo("America/New_York")
COST = 0.0030
RACE_T = 0.02
H1M = 60   # fwd window on 1m
H5M = 12   # fwd window on 5m

EV = pd.read_csv(os.path.join(HERE, "gapper_events.csv"))
EV["key"] = list(zip(EV["symbol"], EV["date"]))
meta = EV.set_index("key")[["gap", "prev_close", "rel_vol"]].to_dict("index")

print("loading event bars...", flush=True)
B = pd.read_csv(os.path.join(HERE, "event_bars.csv"))
B["ts"] = pd.to_datetime(B["t"]).dt.tz_convert(NY)
groups = {k: g for k, g in B.groupby(["symbol", "date"])}
print(f"event-days with bars: {len(groups)}", flush=True)


def prep(g):
    g = g.sort_values("ts").set_index("ts")
    g = g.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    pm = g[g.index.time < dtime(9, 30)]
    rth = g[(g.index.time >= dtime(9, 30)) & (g.index.time <= dtime(15, 59))]
    return pm, rth[["open", "high", "low", "close", "volume"]]


def agg5(rth):
    return rth.resample("5min").agg(open=("open", "first"), high=("high", "max"),
                                    low=("low", "min"), close=("close", "last"),
                                    volume=("volume", "sum")).dropna()


def outcome(df, i, side, H):
    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float); l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    if i + 2 >= len(df):
        return None
    e = o[i + 1]
    if e <= 0:
        return None
    end = min(i + 1 + H, len(df) - 1)
    sgn = 1 if side == "long" else -1
    drift = (c[end] / e - 1) * sgn - COST
    up = e * (1 + RACE_T); dn = e * (1 - RACE_T); race = 0
    for j in range(i + 2, end + 1):
        hu = h[j] >= up; du = l[j] <= dn
        if hu and du:
            break
        if hu:
            race = sgn; break
        if du:
            race = -sgn; break
    mfe = (h[i + 2:end + 1].max() / e - 1) if sgn > 0 else (1 - l[i + 2:end + 1].min() / e)
    return drift, race, mfe


rows = []
for key, g in groups.items():
    m = meta.get(key)
    if m is None:
        continue
    pm, rth = prep(g)
    if len(rth) < 90:
        continue
    pmh = pm["high"].max() if len(pm) >= 10 else np.nan
    r5 = agg5(rth)
    year = key[1][:4]
    ctx = dict(gap=m["gap"], px=m["prev_close"], rv=m["rel_vol"], year=year)

    # 1. gap-and-go: first 1m bar (within first 60 min) whose high breaks premarket high
    if not np.isnan(pmh):
        hh = rth["high"].to_numpy(float)
        for i in range(0, min(60, len(rth))):
            if hh[i] > pmh:
                r = outcome(rth, i, "long", H1M)
                if r:
                    rows.append(dict(pattern="gap_and_go_pmh_break", side="long",
                                     drift=r[0], race=r[1], mfe=r[2], **ctx))
                break
    # 2/3. flat-top + failed flat-top on 1m
    try:
        for s in flat_top_breakouts(rth, min_touches=3, tolerance_pct=0.002)[:6]:
            r = outcome(rth, s.signal_index, "long", H1M)
            if r:
                rows.append(dict(pattern="flat_top_1m", side="long", drift=r[0], race=r[1], mfe=r[2], **ctx))
        for s in failed_flat_top_breakouts(rth, min_touches=3, tolerance_pct=0.002)[:6]:
            r = outcome(rth, s.signal_index, "short", H1M)
            if r:
                rows.append(dict(pattern="failed_flat_top_1m", side="short", drift=r[0], race=r[1], mfe=r[2], **ctx))
    except Exception:
        pass
    # 4. 5-candle reversal on 5m
    try:
        for s in consecutive_candle_reversals(r5, n_consecutive=5)[:4]:
            r = outcome(r5, s.signal_index, s.side, H5M)
            if r:
                rows.append(dict(pattern="rev5_5m", side=s.side, drift=r[0], race=r[1], mfe=r[2], **ctx))
    except Exception:
        pass
    # 5. 10-candle reversal on 1m
    try:
        for s in consecutive_candle_reversals(rth, n_consecutive=10)[:4]:
            r = outcome(rth, s.signal_index, s.side, H1M)
            if r:
                rows.append(dict(pattern="rev10_1m", side=s.side, drift=r[0], race=r[1], mfe=r[2], **ctx))
    except Exception:
        pass

E = pd.DataFrame(rows)
E.to_csv(os.path.join(HERE, "smallcap_results.csv"), index=False)
print(f"\ntotal signals: {len(E)} across {len(groups)} event-days\n")
print(f"{'pattern':<26s}{'side':<7s}{'n':>6s}{'avg net':>9s}{'med':>8s}{'race+':>7s}{'medMFE':>8s}")
for (p, sd), S in E.groupby(["pattern", "side"]):
    w = (S.race == 1).sum(); lo = (S.race == -1).sum()
    print(f"{p:<26s}{sd:<7s}{len(S):>6d}{S.drift.mean()*100:>+8.2f}%{S.drift.median()*100:>+7.2f}%"
          f"{w/max(w+lo,1):>7.0%}{S.mfe.median()*100:>7.2f}%")
print("\nby year (avg net drift %):")
piv = E.pivot_table(index=["pattern", "side"], columns="year", values="drift", aggfunc="mean") * 100
print(piv.round(2).to_string())
print("\nby gap bucket:")
E["gapb"] = pd.cut(E.gap, [0.07, 0.12, 0.25, 5], labels=["7-12%", "12-25%", ">25%"])
piv2 = E.pivot_table(index=["pattern", "side"], columns="gapb", values="drift", aggfunc=["mean", "count"], observed=True)
print((piv2["mean"] * 100).round(2).to_string())
print((piv2["count"]).to_string())
