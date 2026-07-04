"""Part A: replicate R1-R4 scan on QQQ (out-of-sample for the alignment finding).
Part B: stop/target exit grid on the filtered rules, SPY and QQQ:
   R2f long  = R2 + 1h %B < 0.3
   R4f short = R4 + daily %B <= 0.8   (variant: also 30m RSI < 40)
Slippage 0.02%/side. Same-bar stop+target -> stop assumed first (conservative).
"""
import math
import os
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
NY = ZoneInfo("America/New_York")
SLIP = 0.0002
WARM = np.datetime64("2016-06-01")


def wilder_rsi(close, period=14):
    d = close.diff()
    ag = d.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    al = (-d).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


class Sym:
    def __init__(self, sym):
        df = pd.read_csv(os.path.join(OUT, f"{sym}_5m_full.csv"))
        df["ts"] = pd.to_datetime(df["timestamps"]).dt.tz_convert(NY)
        df = df.sort_values("ts").reset_index(drop=True)
        d5 = df[(df["ts"].dt.time >= dtime(9, 30)) & (df["ts"].dt.time <= dtime(15, 55))].reset_index(drop=True)
        c = d5["close"]
        d5["rsi14"] = wilder_rsi(c)
        d5["ema9"] = c.ewm(span=9, adjust=False).mean()
        d5["sma9"] = c.rolling(9).mean()
        mid = c.rolling(20).mean()
        sd = c.rolling(20).std()
        d5["bb_lo"] = mid - 2 * sd
        d5["day"] = d5["ts"].dt.date
        pv = d5["vwap"] * d5["volume"]
        d5["svwap"] = pv.groupby(d5["day"]).cumsum() / d5["volume"].groupby(d5["day"]).cumsum()
        d5["vol20"] = d5["volume"].rolling(20).mean()
        self.d5 = d5
        self.ts = d5["ts"].values
        self.o = d5["open"].values
        self.h = d5["high"].values
        self.l = d5["low"].values
        self.c = d5["close"].values
        self.vw = d5["svwap"].values
        self.ema9 = d5["ema9"].values
        self.sma9 = d5["sma9"].values
        self.rsi = d5["rsi14"].values
        self.bblo = d5["bb_lo"].values
        self.volx = (d5["volume"] / d5["vol20"]).values
        self.day = d5["day"].values
        self.tod_end = (d5["ts"].dt.hour * 60 + d5["ts"].dt.minute + 5).values
        self.n = len(d5)
        self.day_start, self.day_end = {}, {}
        for i, dy in enumerate(self.day):
            if dy not in self.day_start:
                self.day_start[dy] = i
            self.day_end[dy] = i
        # higher TFs
        self.tf = {}
        for step, name in [(15, "15m"), (30, "30m"), (60, "1h")]:
            minutes = (d5["ts"].dt.hour * 60 + d5["ts"].dt.minute) - 570
            grp = np.minimum(minutes // step, (390 // step) - 1)
            key = d5["day"].astype(str) + "_" + grp.astype(str)
            g = d5.groupby(key, sort=False)
            a = pd.DataFrame({"ts": g["ts"].first(), "close": g["close"].last(),
                              "end": g["ts"].last() + pd.Timedelta(minutes=5)})
            a = a.reset_index(drop=True).sort_values("ts").reset_index(drop=True)
            a["rsi14"] = wilder_rsi(a["close"])
            m = a["close"].rolling(20).mean()
            s = a["close"].rolling(20).std()
            a["pctb"] = (a["close"] - (m - 2 * s)) / (4 * s)
            self.tf[name] = (a["end"].values, a[["rsi14", "pctb"]].values)
        daily = pd.read_csv(os.path.join(OUT, f"{sym}_daily.csv"), parse_dates=["date"])
        dc = daily["close"]
        dm = dc.rolling(20).mean()
        dsd = dc.rolling(20).std()
        daily["pctb"] = (dc - (dm - 2 * dsd)) / (4 * dsd)
        self.d_dates = daily["date"].values
        self.d_pctb = daily["pctb"].values

    def tf_at(self, sig_ts, name, col):
        ends, feats = self.tf[name]
        k = np.searchsorted(ends, sig_ts, side="right") - 1
        return feats[k][col] if k >= 0 else np.nan

    def daily_pctb_prev(self, sig_ts):
        dayn = pd.Timestamp(sig_ts).normalize()
        kd = np.searchsorted(self.d_dates, np.datetime64(dayn)) - 1
        return self.d_pctb[kd] if kd >= 0 else np.nan


def scan(S):
    sigs = {"R1_fade": [], "R2_long": [], "R3_fade": [], "R4_cont": []}
    cur_day, touch, in_t = None, 0, False
    last = {k: -99 for k in sigs}
    for i in range(1, S.n):
        if S.ts[i] < WARM or math.isnan(S.rsi[i]) or math.isnan(S.sma9[i]):
            continue
        dy = S.day[i]
        if dy != cur_day:
            cur_day, touch, in_t = dy, 0, False
        if S.rsi[i] >= 65 and not in_t:
            touch += 1
            in_t = True
        elif S.rsi[i] < 58:
            in_t = False
        dsi = S.day_start[dy]
        day_hi = S.h[dsi:i + 1].max()
        px = S.c[i]
        rl = math.ceil(day_hi / 5) * 5
        if (touch >= 2 and in_t and S.rsi[i] >= 65 and px > S.vw[i]
                and 0 < rl - day_hi <= 0.0015 * px and i - last["R1_fade"] > 12):
            sigs["R1_fade"].append(i); last["R1_fade"] = i
        if i - dsi == 5:
            o1, c1, c2 = S.o[dsi], S.c[dsi + 2], S.c[i]
            if c1 < o1 and c2 > o1 and c2 > S.vw[i]:
                sigs["R2_long"].append(i)
        spread = abs(S.ema9[i] - S.sma9[i])
        above = min(S.ema9[i], S.sma9[i]) - S.vw[i]
        crossed = S.ema9[i] < S.sma9[i] and S.ema9[i - 1] >= S.sma9[i - 1]
        if (crossed and spread < 0.0003 * px and 0 < above < 0.0015 * px
                and px > S.vw[i] and px < day_hi * (1 - 0.002) and i - last["R3_fade"] > 12):
            sigs["R3_fade"].append(i); last["R3_fade"] = i
        if (px < S.vw[i] and px < S.bblo[i] and S.volx[i] >= 2.5 and i - last["R4_cont"] > 12):
            sigs["R4_cont"].append(i); last["R4_cont"] = i
    return sigs


def eod_ret(S, i, side):
    if i + 2 >= S.n or S.day[i + 1] != S.day[i]:
        return None
    e = S.o[i + 1]
    de = S.day_end[S.day[i]]
    return side * (S.c[de] / e - 1)


def stop_target(S, i, side, stop, target):
    """Entry next 5m open with slippage; conservative same-bar rule; EOD close."""
    if i + 2 >= S.n or S.day[i + 1] != S.day[i]:
        return None
    e = S.o[i + 1] * (1 + SLIP * side)
    de = S.day_end[S.day[i]]
    if side > 0:
        stop_px, tgt_px = e * (1 - stop), e * (1 + target)
    else:
        stop_px, tgt_px = e * (1 + stop), e * (1 - target)
    for j in range(i + 1, de + 1):
        if side > 0:
            if S.l[j] <= stop_px:
                return (stop_px * (1 - SLIP)) / e - 1
            if S.h[j] >= tgt_px:
                return (tgt_px * (1 - SLIP)) / e - 1
        else:
            if S.h[j] >= stop_px:
                return -((stop_px * (1 + SLIP)) / e - 1)
            if S.l[j] <= tgt_px:
                return -((tgt_px * (1 + SLIP)) / e - 1)
    x = S.c[de]
    return side * (x * (1 - SLIP * side) / e - 1)


for symname in ["QQQ"]:
    S = Sym(symname)
    sigs = scan(S)
    print(f"===== {symname}: rule replication (EOD outcome, no exits overlay) =====")
    for rule, side in [("R1_fade", -1), ("R2_long", +1), ("R3_fade", -1), ("R4_cont", -1)]:
        rows = []
        for i in sigs[rule]:
            r = eod_ret(S, i, side)
            if r is None:
                continue
            rows.append((i, r))
        a = np.array([r for _, r in rows])
        if not len(a):
            continue
        print(f"{rule:<9s} n={len(a):>4d}  EOD WR={(a>0).mean():>6.1%}  avg={a.mean():+.3%}")
        # key companion split
        if rule == "R2_long":
            f = np.array([S.tf_at(S.ts[i], "1h", 1) < 0.3 for i, _ in rows])
        elif rule in ("R3_fade", "R4_cont"):
            f = np.array([S.daily_pctb_prev(S.ts[i]) <= 0.8 for i, _ in rows])
        else:
            f = np.array([S.tf_at(S.ts[i], "30m", 0) > 65 for i, _ in rows])
        for lbl, m in [("aligned", f), ("not", ~f)]:
            b = a[m]
            if len(b) >= 10:
                print(f"    {lbl:<8s} n={len(b):>4d}  WR={(b>0).mean():>6.1%}  avg={b.mean():+.3%}")

print("\n===== Part B: stop/target grids on filtered rules =====")
for symname in ["SPY", "QQQ"]:
    S = Sym(symname)
    sigs = scan(S)
    r2f = [i for i in sigs["R2_long"] if S.tf_at(S.ts[i], "1h", 1) < 0.3]
    r4f = [i for i in sigs["R4_cont"] if S.daily_pctb_prev(S.ts[i]) <= 0.8]
    r4ff = [i for i in r4f if S.tf_at(S.ts[i], "30m", 0) < 40]
    for label, idxs, side in [(f"{symname} R2f long (1h%B<0.3)", r2f, +1),
                              (f"{symname} R4f short (d%B<=.8)", r4f, -1),
                              (f"{symname} R4ff short (+30mRSI<40)", r4ff, -1)]:
        print(f"\n--- {label}: {len(idxs)} signals ---")
        print(f"    {'stop':>6s} {'tgt':>6s} {'n':>5s} {'WR':>6s} {'avg':>8s} {'PF':>5s} {'sum':>7s}")
        for stop in [0.0015, 0.0025]:
            for tgt in [0.002, 0.003, 0.005]:
                rets = [stop_target(S, i, side, stop, tgt) for i in idxs]
                a = np.array([r for r in rets if r is not None])
                if not len(a):
                    continue
                wins, losses = a[a > 0], a[a <= 0]
                pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
                print(f"    {stop:>6.2%} {tgt:>6.2%} {len(a):>5d} {(a>0).mean():>6.1%} "
                      f"{a.mean():>8.3%} {pf:>5.2f} {a.sum():>7.1%}")
