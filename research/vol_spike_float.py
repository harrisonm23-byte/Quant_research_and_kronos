"""Early high-volume directional spike + quiet counter-drift ("float") ->
does the afternoon snap back in the spike's direction?

SPY+QQQ 5m 2016-2026. Volume deseasonalized: vol / median(vol same time slot,
trailing 30 sessions). Spike: first bar 9:35-11:00 with dvol>=3 and |ret|>=0.15%.
Quiet day: from spike+1 to 13:30, >=80% of bars dvol<=1.0 and no bar dvol>=2.0.
Float: sign(close_13:30 - spike close) OPPOSITE to spike direction.
Outcome: 13:30 -> close return, signed POSITIVE = in spike direction.
Baselines: (a) all days 13:30->close |drift|, (b) spike days where drift went
WITH the spike, (c) quiet days with no spike.
"""
import os
from collections import defaultdict, deque
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")


def load(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
    df["day"] = df["ts"].dt.date
    df["slot"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df


def run(sym):
    df = load(sym)
    # deseasonalized volume: vol / trailing-30-session median for that slot
    hist = defaultdict(lambda: deque(maxlen=30))
    dvol = np.full(len(df), np.nan)
    vals = df[["slot", "volume"]].values
    days = df["day"].values
    cur_day = None
    pend = []  # (slot, vol) of current day, appended to hist at day end
    for i, (slot, vol) in enumerate(vals):
        if days[i] != cur_day:
            for s, v in pend:
                hist[s].append(v)
            pend = []
            cur_day = days[i]
        h = hist[slot]
        if len(h) >= 15:
            dvol[i] = vol / np.median(h)
        pend.append((slot, vol))
    df["dvol"] = dvol

    o = df["open"].values; c = df["close"].values
    slot = df["slot"].values
    ret = np.zeros(len(df))
    ret[1:] = c[1:] / c[:-1] - 1
    first_of_day = np.zeros(len(df), bool)
    first_of_day[0] = True
    first_of_day[1:] = days[1:] != days[:-1]
    ret[first_of_day] = (c / o - 1)[first_of_day]

    rows_of = {}
    for i, dy in enumerate(days):
        rows_of.setdefault(dy, []).append(i)

    CHK = 13 * 60 + 30  # 13:30
    ev = []
    all_pm = []          # baseline: every day's 13:30->close signed-random
    for dy, idxs in rows_of.items():
        idxs = np.array(idxs)
        dv = df["dvol"].values[idxs]
        if np.isnan(dv).all() or len(idxs) < 70:
            continue
        # checkpoint index
        chk_pos = np.where(slot[idxs] >= CHK)[0]
        if len(chk_pos) == 0:
            continue
        ichk = idxs[chk_pos[0]]
        iend = idxs[-1]
        pm = c[iend] / c[ichk] - 1
        all_pm.append(pm)
        # find first early spike (9:35-11:00)
        spike = None
        for j, i in enumerate(idxs):
            if slot[i] < 575 or slot[i] > 660:   # 9:35..11:00
                continue
            if not np.isnan(dv[j]) and dv[j] >= 3.0 and abs(ret[i]) >= 0.0015:
                spike = (j, i)
                break
        if spike is None:
            continue
        j, i = spike
        sdir = 1 if ret[i] > 0 else -1
        # quiet from spike+1 to checkpoint
        mid = idxs[(idxs > i) & (idxs < ichk)]
        if len(mid) < 12:
            continue
        dmid = df["dvol"].values[mid]
        dmid = dmid[~np.isnan(dmid)]
        if len(dmid) == 0:
            continue
        quiet = (dmid <= 1.0).mean() >= 0.80 and dmid.max() < 2.0
        drift = (c[ichk] - c[i]) / c[i]                 # spike close -> 13:30
        pm_signed = pm * sdir                            # + = spike direction
        drift_signed = drift * sdir
        # did afternoon contain its own volume spike?
        aft = idxs[idxs > ichk]
        davt = df["dvol"].values[aft]
        pm_spike = np.nanmax(davt) >= 2.0 if len(aft) else False
        # did price revisit the spike bar extreme?
        lo = df["low"].values; hi = df["high"].values
        if sdir < 0:
            revisit = (lo[aft] <= lo[i]).any() if len(aft) else False
        else:
            revisit = (hi[aft] >= hi[i]).any() if len(aft) else False
        ev.append(dict(day=dy, sdir=sdir, quiet=quiet, drift_signed=drift_signed,
                       pm_signed=pm_signed, pm_spike=pm_spike, revisit=revisit,
                       spike_ret=ret[i] * sdir,
                       half=1 if str(dy) < "2021-07-01" else 2))
    E = pd.DataFrame(ev)
    ap = np.array(all_pm)
    print(f"\n================ {sym} ================")
    print(f"spike days: {len(E):,} ({len(E)/10.1:.0f}/yr) | baseline 13:30->close: mean {ap.mean()*100:+.3f}% P(up) {(ap>0).mean():.0%}")

    def rep(lbl, S):
        if len(S) < 25:
            print(f"  {lbl:<44s} n={len(S)} (too few)")
            return
        a = S["pm_signed"].values
        print(f"  {lbl:<44s} n={len(S):>4d}  PM in spike dir: {(a>0).mean():.0%}  avg {a.mean()*100:+.3f}%"
              f"  revisit extreme {S['revisit'].mean():.0%}  own PM spike {S['pm_spike'].mean():.0%}")

    # the user's exact scenario: quiet day + float AGAINST spike
    rep("USER CASE: quiet + float AGAINST spike", E[E.quiet & (E.drift_signed < 0)])
    rep("  .. down-spike, floated UP (short setup)", E[E.quiet & (E.drift_signed < 0) & (E.sdir < 0)])
    rep("  .. up-spike, floated DOWN (long setup)", E[E.quiet & (E.drift_signed < 0) & (E.sdir > 0)])
    rep("control: quiet + float WITH spike", E[E.quiet & (E.drift_signed > 0)])
    rep("control: NOT quiet + float against spike", E[~E.quiet & (E.drift_signed < 0)])
    rep("all spike days", E)
    # halves for the user case
    U = E[E.quiet & (E.drift_signed < 0)]
    for hlf in [1, 2]:
        rep(f"  user case half {hlf}", U[U.half == hlf])
    # size of float matters?
    if len(U) >= 60:
        med = U["drift_signed"].median()
        rep("  user case, small float", U[U.drift_signed >= med])
        rep("  user case, big float (deep retrace)", U[U.drift_signed < med])
    return E


for sym in ["SPY", "QQQ"]:
    run(sym)
