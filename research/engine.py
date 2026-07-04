"""Daily-bar backtesting engine for the 10-strategy suite.

Execution model:
- Signals evaluated at close of day t using completed bars only.
- Entries/exits fill at day t+1 open (exit_fill='open'), or at day t close
  (exit_fill='close', used only for the theoretical same-close variants).
- Slippage: 0.02% per side (buy fills higher, sell fills lower). Commission 0.
- Stop-loss (when enabled): checked intraday from the entry day onward.
  If open gaps below stop -> fill at open; else if low breaches stop -> fill
  at stop price. Slippage applied on top.
- Long-only, one position per run, 100% of strategy equity per trade.
"""
import math
import os

import numpy as np
import pandas as pd

SLIP = 0.0002
STAT_START = pd.Timestamp("2017-04-01")  # all indicators (incl SMA300) warmed up
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def wilder_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rsi = 100 - 100 / (1 + ag / al)
    rsi[al == 0] = 100.0
    return rsi


def load_symbol(sym):
    df = pd.read_csv(os.path.join(DATA_DIR, f"{sym}_daily.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    df["sma5"] = c.rolling(5).mean()
    df["sma10"] = c.rolling(10).mean()
    df["sma200"] = c.rolling(200).mean()
    df["sma300"] = c.rolling(300).mean()
    df["rsi2"] = wilder_rsi(c, 2)
    df["cumrsi2"] = df["rsi2"] + df["rsi2"].shift(1)
    df["rsi5"] = wilder_rsi(c, 5)
    df["rsi5_1"] = df["rsi5"].shift(1)
    df["rsi5_2"] = df["rsi5"].shift(2)
    df["rsi5_3"] = df["rsi5"].shift(3)
    df["rsi21"] = wilder_rsi(c, 21)
    rng = (h - l)
    df["ibs"] = np.where(rng > 0, (c - l) / rng, 0.5)
    df["lc7"] = c.rolling(7).min()
    df["hc7"] = c.rolling(7).max()
    df["lc5"] = c.rolling(5).min()
    df["ll5"] = l.rolling(5).min()
    df["hh10"] = h.rolling(10).max()
    df["avg_range25"] = rng.rolling(25).mean()
    df["lower_band"] = df["hh10"] - 2.5 * df["avg_range25"]
    df["ret1"] = c / c.shift(1) - 1
    df["prev_close"] = c.shift(1)
    df["prev2_close"] = c.shift(2)
    df["prev_high"] = h.shift(1)
    df["prev_low"] = l.shift(1)
    df["weekday"] = df["date"].dt.weekday  # 0 = Monday
    return df


def run_bt(df, entry_fn, exit_fn=None, stop_pct=None, max_hold=None,
           exit_fill="open", entry_fill="open", regime_fn=None, slip=SLIP,
           stat_start=STAT_START):
    """Event-driven daily backtest. Returns (equity Series, trades DataFrame)."""
    dates = df["date"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    rows = list(df.itertuples(index=False))
    n = len(df)

    equity = 1.0
    shares = 0.0
    in_pos = False
    pending_entry = False
    pending_exit = False
    exit_reason = None
    entry_px = entry_date = None
    hold = 0
    eq = np.full(n, np.nan)
    trades = []

    start_i = int(np.searchsorted(dates, np.datetime64(stat_start)))

    def close_trade(i, px, reason):
        nonlocal equity, in_pos, shares, pending_exit, hold
        equity = shares * px
        trades.append(dict(entry_date=pd.Timestamp(entry_date), exit_date=pd.Timestamp(dates[i]),
                           entry_px=entry_px, exit_px=px, ret=px / entry_px - 1,
                           hold_days=hold, reason=reason))
        in_pos = False
        shares = 0.0
        pending_exit = False
        hold = 0

    for i in range(start_i, n):
        # --- open: fills from yesterday's signals ---
        if in_pos and pending_exit:
            close_trade(i, o[i] * (1 - slip), exit_reason)
        elif (not in_pos) and pending_entry:
            entry_px = o[i] * (1 + slip)
            entry_date = dates[i]
            shares = equity / entry_px
            in_pos = True
            hold = 0
        pending_entry = False

        # --- intraday: stop loss ---
        if in_pos and stop_pct is not None:
            stop_px = entry_px * (1 - stop_pct)
            if o[i] <= stop_px:
                close_trade(i, o[i] * (1 - slip), "stop_gap")
            elif l[i] <= stop_px:
                close_trade(i, stop_px * (1 - slip), "stop")

        # --- close: mark equity, evaluate signals ---
        if in_pos:
            hold += 1
        r = rows[i]
        if in_pos:
            want_exit = False
            reason = None
            if exit_fn is not None and exit_fn(r):
                want_exit, reason = True, "signal"
            if not want_exit and max_hold is not None and hold >= max_hold:
                want_exit, reason = True, "time"
            if not want_exit and regime_fn is not None and regime_fn(r):
                want_exit, reason = True, "regime"
            if want_exit:
                if exit_fill == "close":
                    close_trade(i, c[i] * (1 - slip), reason)
                else:
                    pending_exit = True
                    exit_reason = reason
        elif not pending_entry:
            blocked = regime_fn is not None and regime_fn(r)
            if not blocked and entry_fn(r):
                if entry_fill == "close":
                    entry_px = c[i] * (1 + slip)
                    entry_date = dates[i]
                    shares = equity / entry_px
                    in_pos = True
                    hold = 0
                else:
                    pending_entry = True

        eq[i] = shares * c[i] if in_pos else equity

    eq_series = pd.Series(eq[start_i:], index=pd.DatetimeIndex(dates[start_i:]), name="equity")
    return eq_series, pd.DataFrame(trades)


def compute_stats(eq, trades, label=""):
    days = (eq.index[-1] - eq.index[0]).days
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total) ** (365.25 / days) - 1 if days > 0 else 0.0
    dr = eq.pct_change().dropna()
    sharpe = dr.mean() / dr.std() * math.sqrt(252) if dr.std() > 0 else 0.0
    dn = dr[dr < 0]
    sortino = dr.mean() / dn.std() * math.sqrt(252) if len(dn) > 1 and dn.std() > 0 else 0.0
    peak = eq.cummax()
    maxdd = ((eq - peak) / peak).min()
    ntr = len(trades)
    if ntr:
        rets = trades["ret"].values
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        wr = len(wins) / ntr
        pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf")
        avg_tr, med_tr = rets.mean(), float(np.median(rets))
        avg_hold = trades["hold_days"].mean()
        best, worst = rets.max(), rets.min()
        in_mkt_days = trades["hold_days"].sum()
    else:
        wr = pf = avg_tr = med_tr = avg_hold = best = worst = in_mkt_days = 0.0
    exposure = in_mkt_days / len(eq) if len(eq) else 0.0
    yearly = eq.resample("YE").last().pct_change()
    first_year = eq.resample("YE").last()
    if len(first_year) > 0:
        yearly.iloc[0] = first_year.iloc[0] / eq.iloc[0] - 1
    return dict(label=label, cagr=cagr, total=total, maxdd=maxdd, sharpe=sharpe,
                sortino=sortino, wr=wr, pf=pf, avg_trade=avg_tr, med_trade=med_tr,
                avg_hold=avg_hold, n_trades=ntr, exposure=exposure,
                best=best, worst=worst,
                yearly={str(k.year): float(v) for k, v in yearly.items() if not math.isnan(v)})


SUBPERIODS = [
    ("2000-2007", "2000-01-01", "2008-01-01"),
    ("2008-2009", "2008-01-01", "2010-01-01"),
    ("2010-2019", "2010-01-01", "2020-01-01"),
    ("2020-2021", "2020-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023-present", "2023-01-01", "2027-01-01"),
]


def subperiod_stats(eq, trades):
    out = {}
    for name, a, b in SUBPERIODS:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        sl = eq[(eq.index >= a) & (eq.index < b)]
        if len(sl) < 20:
            out[name] = None
            continue
        tr = trades[(trades["exit_date"] >= a) & (trades["exit_date"] < b)] if len(trades) else trades
        out[name] = compute_stats(sl, tr, name)
    return out
