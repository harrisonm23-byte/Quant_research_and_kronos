
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


Side = Literal["long", "short"]


@dataclass(frozen=True)
class Signal:
    pattern: str
    side: Side
    signal_index: int
    trigger: float
    reference_level: float
    metadata: dict


def _validate_ohlcv(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted ascending")


def consecutive_candle_reversals(
    df: pd.DataFrame,
    *,
    n_consecutive: int,
    tick_size: float = 0.01,
) -> list[Signal]:
    """
    Guide-faithful reversal trigger:
      - Long: after >= n consecutive red candles, the first subsequent candle
        whose high exceeds the immediately prior candle's high.
      - Short: after >= n consecutive green candles, the first subsequent candle
        whose low falls below the immediately prior candle's low.

    This mirrors the guide's:
      - 5-minute: first candle to make a new high after 5 consecutive red candles.
      - 1-minute: first candle to make a new high after 10 consecutive red candles.
    """
    _validate_ohlcv(df)
    if n_consecutive < 2:
        raise ValueError("n_consecutive must be >= 2")

    red = (df["close"] < df["open"]).to_numpy()
    green = (df["close"] > df["open"]).to_numpy()
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)

    signals: list[Signal] = []
    red_run = 0
    green_run = 0
    long_armed = False
    short_armed = False

    for i in range(len(df)):
        if red[i]:
            red_run += 1
            green_run = 0
        elif green[i]:
            green_run += 1
            red_run = 0
        else:
            red_run = 0
            green_run = 0

        if red_run >= n_consecutive:
            long_armed = True
        if green_run >= n_consecutive:
            short_armed = True

        if i == 0:
            continue

        if long_armed and highs[i] > highs[i - 1]:
            signals.append(
                Signal(
                    pattern=f"{n_consecutive}_candle_reversal",
                    side="long",
                    signal_index=i,
                    trigger=highs[i - 1] + tick_size,
                    reference_level=highs[i - 1],
                    metadata={"run_length_min": n_consecutive},
                )
            )
            long_armed = False
            red_run = 0

        if short_armed and lows[i] < lows[i - 1]:
            signals.append(
                Signal(
                    pattern=f"{n_consecutive}_candle_reversal",
                    side="short",
                    signal_index=i,
                    trigger=lows[i - 1] - tick_size,
                    reference_level=lows[i - 1],
                    metadata={"run_length_min": n_consecutive},
                )
            )
            short_armed = False
            green_run = 0

    return signals


def flat_top_breakouts(
    df: pd.DataFrame,
    *,
    min_touches: int = 3,
    formation_min_bars: int = 4,
    formation_max_bars: int = 20,
    tolerance_pct: float = 0.001,
    tick_size: float = 0.01,
) -> list[Signal]:
    """
    Objective implementation of the guide's "buy first candle that breaks flat top."

    The guide supplies the entry concept, but not numeric formation tolerances.
    Therefore min_touches, lookback, and tolerance are explicit research parameters,
    not claims about the guide.
    """
    _validate_ohlcv(df)
    if min_touches < 2:
        raise ValueError("min_touches must be >= 2")
    if not 0 < tolerance_pct < 0.05:
        raise ValueError("tolerance_pct must be between 0 and 5%")

    highs = df["high"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    signals: list[Signal] = []
    last_breakout_level: Optional[float] = None

    for i in range(formation_min_bars, len(df)):
        found = None
        max_window = min(formation_max_bars, i)
        for length in range(formation_min_bars, max_window + 1):
            start = i - length
            window_highs = highs[start:i]
            level = float(np.max(window_highs))
            tol = level * tolerance_pct
            touches = int(np.sum(np.abs(window_highs - level) <= tol))

            if touches < min_touches:
                continue
            if np.any(closes[start:i] > level + tol):
                continue
            if highs[i] <= level + tick_size:
                continue

            found = (level, touches, length)
            break

        if found is None:
            continue

        level, touches, length = found
        # Avoid repeated signals from the same level cluster.
        if last_breakout_level is not None and abs(level - last_breakout_level) <= level * tolerance_pct:
            continue

        signals.append(
            Signal(
                pattern="flat_top_breakout",
                side="long",
                signal_index=i,
                trigger=level + tick_size,
                reference_level=level,
                metadata={
                    "touches": touches,
                    "formation_bars": length,
                    "tolerance_pct": tolerance_pct,
                },
            )
        )
        last_breakout_level = level

    return signals


def failed_flat_top_breakouts(
    df: pd.DataFrame,
    *,
    min_touches: int = 3,
    formation_min_bars: int = 4,
    formation_max_bars: int = 20,
    tolerance_pct: float = 0.001,
    failure_window: int = 3,
    tick_size: float = 0.01,
) -> list[Signal]:
    """
    Research operationalization of the guide's false-breakout / bull-trap examples.

    This is deliberately labeled an operationalization because the guide illustrates
    the trap visually but does not provide a fully mechanical entry rule.

    Rule:
      1. Detect a valid flat-top level.
      2. Price trades above it.
      3. Within failure_window bars, a candle closes back below the level.
      4. Short trigger is one tick below that failure candle's low.
    """
    _validate_ohlcv(df)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    signals: list[Signal] = []

    for i in range(formation_min_bars, len(df)):
        max_window = min(formation_max_bars, i)
        candidate = None
        for length in range(formation_min_bars, max_window + 1):
            start = i - length
            window_highs = highs[start:i]
            level = float(np.max(window_highs))
            tol = level * tolerance_pct
            touches = int(np.sum(np.abs(window_highs - level) <= tol))
            if touches >= min_touches and not np.any(closes[start:i] > level + tol):
                if highs[i] > level + tick_size:
                    candidate = (level, touches, length)
                    break
        if candidate is None:
            continue

        level, touches, length = candidate
        end = min(len(df), i + failure_window + 1)
        for j in range(i, end):
            if closes[j] < level:
                signals.append(
                    Signal(
                        pattern="failed_flat_top_breakout",
                        side="short",
                        signal_index=j,
                        trigger=lows[j] - tick_size,
                        reference_level=level,
                        metadata={
                            "breakout_index": i,
                            "failure_bars": j - i,
                            "touches": touches,
                            "formation_bars": length,
                        },
                    )
                )
                break

    # De-duplicate overlapping traps around the same level and time.
    deduped: list[Signal] = []
    for s in signals:
        if deduped:
            prev = deduped[-1]
            if (
                s.signal_index - prev.signal_index <= failure_window
                and abs(s.reference_level - prev.reference_level)
                <= s.reference_level * tolerance_pct
            ):
                continue
        deduped.append(s)
    return deduped


def simulate_intraday_barrier_trade(
    df: pd.DataFrame,
    signal: Signal,
    *,
    stop_r: float = 1.0,
    target_r: float = 2.0,
    max_hold_bars: int = 12,
    initial_risk_pct: float = 0.0025,
    slippage_bps: float = 1.0,
) -> dict:
    """
    Conservative OHLC-bar simulator for signal utility.

    Entry:
      - Triggered no earlier than the signal bar.
      - Long fill = trigger plus slippage; short fill = trigger minus slippage.
    Stop/target:
      - Expressed in R using an explicit percentage risk budget.
      - If stop and target are both touched in the same bar, stop wins
        (conservative sequencing).
    """
    _validate_ohlcv(df)
    i = signal.signal_index
    if i >= len(df):
        raise IndexError("signal_index outside DataFrame")

    slip = slippage_bps / 10_000
    if signal.side == "long":
        entry = signal.trigger * (1 + slip)
        risk = entry * initial_risk_pct
        stop = entry - stop_r * risk
        target = entry + target_r * risk
    else:
        entry = signal.trigger * (1 - slip)
        risk = entry * initial_risk_pct
        stop = entry + stop_r * risk
        target = entry - target_r * risk

    exit_index = min(len(df) - 1, i + max_hold_bars)
    exit_price = float(df["close"].iloc[exit_index])
    reason = "time"

    for j in range(i, min(len(df), i + max_hold_bars + 1)):
        high = float(df["high"].iloc[j])
        low = float(df["low"].iloc[j])

        if signal.side == "long":
            stop_hit = low <= stop
            target_hit = high >= target
            if stop_hit:
                exit_index, exit_price, reason = j, stop, "stop"
                break
            if target_hit:
                exit_index, exit_price, reason = j, target, "target"
                break
        else:
            stop_hit = high >= stop
            target_hit = low <= target
            if stop_hit:
                exit_index, exit_price, reason = j, stop, "stop"
                break
            if target_hit:
                exit_index, exit_price, reason = j, target, "target"
                break

    pnl_pct = (
        (exit_price / entry - 1)
        if signal.side == "long"
        else (entry / exit_price - 1)
    )
    return {
        "pattern": signal.pattern,
        "side": signal.side,
        "entry_index": i,
        "exit_index": exit_index,
        "entry": entry,
        "exit": exit_price,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "hold_bars": exit_index - i + 1,
    }
