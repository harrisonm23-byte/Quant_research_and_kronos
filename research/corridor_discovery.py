"""DISCOVERY MODE (not the gauntlet): the SMA9 / lower-Bollinger corridor.

The zone the user named: close BELOW SMA9 but ABOVE the lower Bollinger band —
"weak but not broken", the band price rides before it releases.

We MAP, we do not judge:
  - forward MFE (max favorable) / MAE (max adverse) over 6 & 12 bars, let winners run
  - median favorable move, P(reach +0.2/0.3/0.5%) up AND down
  - response surface across: dwell (bars coiling in zone) x compression (BB width),
    position in corridor (near SMA9 vs near lower BB), time of day
  - RELEASE events: after coiling, first close back above SMA9 (up-release) vs
    first close below lower BB (down-break) -> forward distribution
Baselines from all bars. SPY + QQQ 5m, both halves shown (not used to kill).
BB = SMA20 +/- 2*std(ddof=0) to match charting platforms. SMA9 = 9-SMA.
"""
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
H1, H2 = 6, 12  # 30 / 60 min


def wilder_rsi(c, n=14):
    d = np.diff(c, prepend=c[0]); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = np.zeros_like(c); ad = np.zeros_like(c); au[n] = up[1:n+1].mean(); ad[n] = dn[1:n+1].mean()
    for i in range(n+1, len(c)):
        au[i] = (au[i-1]*(n-1)+up[i])/n; ad[i] = (ad[i-1]*(n-1)+dn[i])/n
    rs = np.divide(au, ad, out=np.full_like(c, np.inf), where=ad > 0)
    return 100 - 100/(1+rs)


def load(sym):
    df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
    df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
    df = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].sort_values("ts").reset_index(drop=True)
    df["day"] = df["ts"].dt.date
    c = df["close"]
    df["sma9"] = c.rolling(9).mean()
    df["sma20"] = c.rolling(20).mean()
    sd = c.rolling(20).std(ddof=0)
    df["lbb"] = df["sma20"] - 2*sd
    df["ubb"] = df["sma20"] + 2*sd
    df["bbw"] = (df["ubb"] - df["lbb"]) / df["sma20"]
    df["rsi"] = wilder_rsi(c.values)
    pv = df["vwap"] * df["volume"]
    df["svwap"] = pv.groupby(df["day"]).cumsum() / df["volume"].groupby(df["day"]).cumsum()
    return df


def fwd(i, H, c, h, l, de):
    """forward MFE up, MAE dn, drift, over min(H, to end of day)."""
    end = min(i + H, de)
    if end <= i:
        return np.nan, np.nan, np.nan
    seg_h = h[i+1:end+1]; seg_l = l[i+1:end+1]
    mfe = seg_h.max()/c[i] - 1
    mae = seg_l.min()/c[i] - 1
    drift = c[end]/c[i] - 1
    return mfe, mae, drift


def pct(a):
    return np.array(a)


def run(sym):
    df = load(sym)
    c = df["close"].values; h = df["high"].values; l = df["low"].values
    s9 = df["sma9"].values; lbb = df["lbb"].values; ubb = df["ubb"].values
    bbw = df["bbw"].values; rsi = df["rsi"].values; vw = df["svwap"].values
    day = df["day"].values; ts = df["ts"].values
    tod = (df["ts"].dt.hour*60 + df["ts"].dt.minute).values
    n = len(df)
    rows_of = {}
    for i, dy in enumerate(day):
        rows_of.setdefault(dy, []).append(i)
    de_of = {dy: idxs[-1] for dy, idxs in rows_of.items()}
    WARM = np.datetime64("2016-06-01")

    # compression terciles (global)
    valid_bbw = bbw[~np.isnan(bbw)]
    q33, q67 = np.nanpercentile(valid_bbw, [33, 67])

    # corridor state + dwell
    in_zone = (~np.isnan(s9)) & (~np.isnan(lbb)) & (s9 > lbb) & (c < s9) & (c > lbb)
    dwell = np.zeros(n, int)
    for i in range(1, n):
        if day[i] == day[i-1] and in_zone[i]:
            dwell[i] = dwell[i-1] + 1 if in_zone[i-1] else 1
        elif in_zone[i]:
            dwell[i] = 1

    # ---- baseline forward dist from all bars ----
    base_mfe = []; base_mae = []; base_drift = []
    for i in range(30, n-1, 3):
        if ts[i] < WARM:
            continue
        de = de_of[day[i]]
        m, a, d = fwd(i, H2, c, h, l, de)
        if not np.isnan(m):
            base_mfe.append(m); base_mae.append(a); base_drift.append(d)
    bmfe, bmae, bdr = pct(base_mfe), pct(base_mae), pct(base_drift)

    # ---- in-zone events ----
    recs = []
    for i in range(30, n-1):
        if ts[i] < WARM or not in_zone[i]:
            continue
        de = de_of[day[i]]
        m, a, d = fwd(i, H2, c, h, l, de)
        if np.isnan(m):
            continue
        posn = (c[i]-lbb[i])/(s9[i]-lbb[i])            # 0=lbb,1=sma9
        comp = "tight" if bbw[i] < q33 else ("wide" if bbw[i] > q67 else "mid")
        dwb = "1-2" if dwell[i] <= 2 else ("3-5" if dwell[i] <= 5 else ("6-10" if dwell[i] <= 10 else "11+"))
        tb = "AM" if tod[i] < 690 else ("MID" if tod[i] < 810 else "PM")
        recs.append(dict(mfe=m, mae=a, drift=d, posn=posn, comp=comp, dwb=dwb, tb=tb,
                         rsi=rsi[i], half=1 if ts[i] < np.datetime64("2021-07-01") else 2))
    E = pd.DataFrame(recs)

    print(f"\n============================ {sym} ============================")
    print(f"in-corridor bars: {len(E):,}  | compression terciles bbw q33={q33:.4f} q67={q67:.4f}")
    print(f"BASELINE (all bars, 60min fwd): MFEup med {np.median(bmfe)*100:.3f}%  MAEdn med {np.median(bmae)*100:.3f}%  "
          f"drift med {np.median(bdr)*100:+.3f}%  P(MFE>=+0.3%) {(bmfe>=0.003).mean():.0%}  P(MAE<=-0.3%) {(bmae<=-0.003).mean():.0%}")

    def summ(lbl, S):
        if len(S) < 150:
            print(f"  {lbl:<22s} n={len(S)} (thin)"); return
        m = S["mfe"].values; a = S["mae"].values; d = S["drift"].values
        print(f"  {lbl:<22s} n={len(S):>6d}  MFEup {np.median(m)*100:5.3f}%  MAEdn {np.median(a)*100:6.3f}%  "
              f"drift {np.median(d)*100:+5.3f}%  P+0.3 {(m>=0.003).mean():4.0%}  P-0.3 {(a<=-0.003).mean():4.0%}  "
              f"skew {(m>=0.003).mean()-(a<=-0.003).mean():+3.0%}")

    print("\n RESPONSE SURFACE — position in corridor (0=lowerBB .. 1=SMA9):")
    summ("near lowerBB (<0.33)", E[E.posn < 0.33])
    summ("middle (0.33-0.67)", E[(E.posn >= 0.33) & (E.posn <= 0.67)])
    summ("near SMA9 (>0.67)", E[E.posn > 0.67])

    print("\n RESPONSE SURFACE — dwell (bars coiling in zone) x compression:")
    for dwb in ["1-2", "3-5", "6-10", "11+"]:
        for comp in ["tight", "mid", "wide"]:
            summ(f"dwell {dwb} / {comp}", E[(E.dwb == dwb) & (E.comp == comp)])

    print("\n RESPONSE SURFACE — time of day:")
    for tb in ["AM", "MID", "PM"]:
        summ(f"{tb}", E[E.tb == tb])

    print("\n TIGHT-COMPRESSION coil (bbw tight, dwell>=6) split by half:")
    coil = E[(E.comp == "tight") & (E.dwb.isin(["6-10", "11+"]))]
    summ("tight coil ALL", coil)
    summ("  half 1", coil[coil.half == 1])
    summ("  half 2", coil[coil.half == 2])

    # ---- RELEASE events ----
    rel = []
    for i in range(31, n-1):
        if ts[i] < WARM:
            continue
        # need to be in corridor with dwell>=3 at bar i-1, then release at i
        if not (in_zone[i-1] and dwell[i-1] >= 3 and day[i] == day[i-1]):
            continue
        de = de_of[day[i]]
        up = c[i] > s9[i-1]           # closed back above prior SMA9 (up-release)
        dn = c[i] < lbb[i-1]          # closed below prior lower BB (down-break)
        if not (up or dn):
            continue
        m, a, d = fwd(i, H2, c, h, l, de)
        if np.isnan(m):
            continue
        comp = "tight" if bbw[i-1] < q33 else ("wide" if bbw[i-1] > q67 else "mid")
        rel.append(dict(dir="UP" if up else "DOWN", mfe=m, mae=a, drift=d, comp=comp,
                        half=1 if ts[i] < np.datetime64("2021-07-01") else 2))
    R = pd.DataFrame(rel)
    print("\n RELEASE from coil (dwell>=3), forward 60min, signed to release dir:")
    def rsumm(lbl, S, sgn):
        if len(S) < 80:
            print(f"  {lbl:<26s} n={len(S)} (thin)"); return
        # signed: favorable = release direction
        fav = S["mfe"].values if sgn > 0 else -S["mae"].values
        adv = -S["mae"].values if sgn > 0 else S["mfe"].values
        dr = S["drift"].values * sgn
        print(f"  {lbl:<26s} n={len(S):>5d}  favMFE {np.median(fav)*100:5.3f}%  advMAE {np.median(adv)*100:5.3f}%  "
              f"drift {np.median(dr)*100:+5.3f}%  P(fav>=+0.3%) {(fav>=0.003).mean():4.0%}")
    if len(R):
        rsumm("UP-release ALL", R[R.dir == "UP"], +1)
        rsumm("  UP tight-comp", R[(R.dir == "UP") & (R.comp == "tight")], +1)
        rsumm("  UP half1", R[(R.dir == "UP") & (R.half == 1)], +1)
        rsumm("  UP half2", R[(R.dir == "UP") & (R.half == 2)], +1)
        rsumm("DOWN-break ALL", R[R.dir == "DOWN"], -1)
        rsumm("  DOWN tight-comp", R[(R.dir == "DOWN") & (R.comp == "tight")], -1)
        n_up = (R.dir == "UP").sum(); n_dn = (R.dir == "DOWN").sum()
        print(f"  release direction base rate: UP {n_up/(n_up+n_dn):.0%} / DOWN {n_dn/(n_up+n_dn):.0%}  (n={n_up+n_dn})")
    return E, R


for sym in ["SPY", "QQQ"]:
    run(sym)
