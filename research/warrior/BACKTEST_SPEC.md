# Warrior Trading Chart-Pattern Backtest: Fidelity Specification

## Purpose

Evaluate whether the chart-pattern concepts in the attached *Chart Pattern Study Guide* contain measurable predictive value without replacing the guide's visual rules with unrelated indicators.

The study has two distinct tracks:

1. **Generic liquid-market translation:** SPY, QQQ, and IWM.
2. **Guide-faithful small-cap momentum study:** a dynamic daily universe of pre-market gappers, using 1-minute and 5-minute bars, extended-hours data, VWAP, pre-market levels, high-of-day levels, and whole/half-dollar levels.

A SPY-only 5-minute test is not treated as a complete test of the guide.

## Rules taken directly from the guide

### Flat-top breakout
- Form a repeated flat resistance level.
- Enter on the **first candle that breaks the flat top**.

Numeric touch counts, tolerance, and formation length are research parameters because the guide does not supply numerical values.

### Bull flag
- After an upward impulse and pullback, price above the relevant pullback candle's high is a buy trigger.
- A subsequent candle close above the prior candle high is the confirmation variant.

Both trigger and close-confirmed versions must be reported separately.

### Bear flag
- After a downward impulse and bounce/consolidation, short the **first candle to make a new low**.

### Five-minute exhaustion reversal
- After at least five consecutive red 5-minute candles, buy the first 5-minute candle to make a new high.
- Mirror the rule after five consecutive green candles for the short-side test.

### One-minute exhaustion reversal
- After at least ten consecutive red 1-minute candles, buy the first 1-minute candle to make a new high.
- Mirror the rule after ten consecutive green candles for the short-side test.

### Guide context retained in the small-cap track
- Pre-market chart and pre-market high
- Gap-and-go and red-to-green
- First pullback on 1-minute and 5-minute charts
- Opening-range breakout
- VWAP break, first pullback after VWAP break, VWAP breakout, and VWAP fade
- High-of-day breakout
- Whole-dollar and half-dollar levels
- Trend-shift short and bear flags
- Halt-related setups only when halt/resumption data are available

## Research operationalizations, not direct guide instructions

The guide illustrates false breakouts and bull traps but does not provide one complete mechanical entry rule. The initial test therefore labels this separately:

1. Detect an objectively valid flat-top level.
2. Price trades above the level.
3. Within one to three bars, price closes back below it.
4. Short trigger is one tick below the failure candle's low.

Alternative definitions are sensitivity tests and cannot be described as the guide's exact rule.

## Data and look-ahead controls

- Source bars: 1-minute OHLCV; aggregate 5-minute bars internally.
- Generic track: regular session only unless a setup explicitly requires pre-market.
- Small-cap track: include pre-market and regular session.
- All rolling levels and indicators use only information available before the trigger.
- No centered pivots or future-confirmed extrema.
- Entry is a stop trigger above/below the specified candle or level.
- When bar data cannot establish intrabar sequence, use the conservative outcome.
- Report a next-bar-open implementation as a separate, more conservative execution variant.
- Add explicit slippage and transaction costs.
- One position per symbol and pattern cluster; no overlapping duplicate entries.

## Phase 1: Signal utility before trade optimization

For every signal, calculate:

- Forward return after 1, 3, 6, 12, and 24 bars
- Maximum favorable excursion
- Maximum adverse excursion
- Probability of reaching symmetric barriers first
- Probability of reaching 0.5R, 1R, 1.5R, and 2R before -1R
- Time of day
- Gap size and relative-volume bucket where applicable
- Trend, VWAP, and volatility regime
- Random-entry baseline matched by symbol, date, and time of day

The first question is whether the signal predicts subsequent price movement. Options are not added until the underlying signal survives this stage.

## Initial detector set

1. Flat-top breakout
2. Failed flat-top breakout / bull trap
3. Five-consecutive-candle 5-minute reversal
4. Ten-consecutive-candle 1-minute reversal
5. Bull and bear flags, after the simpler detectors are validated

## Parameter discipline

Parameters must be narrow and economically interpretable:

- Flat-top touches: 2, 3, or 4
- Level tolerance: 0.05%, 0.10%, or 0.15%
- Formation length: 4-10, 4-20, and 8-30 bars
- Failure window: 1, 2, or 3 bars
- Reversal run length: exact guide threshold and modest stricter variants
- Volume confirmation: none, 1.25x, or 1.5x trailing average
- Entry: intrabar trigger versus next-bar open

The guide-defined threshold is always displayed separately from optimized variants.

## Robustness and keeper rules

- At least 50 trades
- Profit factor at least 1.5 overall
- Profit factor at least 1.2 in each sample half
- Average trade at least 0.15% after costs
- Drawdown shallower than the relevant benchmark
- Results split into 2016-2021 and 2022-2026 where data allow
- Replication on SPY, QQQ, and IWM for generic patterns
- For the small-cap track, report by price, gap, relative volume, float, and year
- Flag any result with fewer than 100 trades for reduced confidence
- Use holdout or walk-forward validation for parameter selection

## Options overlay

Only signals that survive the underlying-price study proceed to options:

- 30-45 DTE, near-ATM or 40-50 delta
- Calls for bullish signals and puts for bearish signals
- Naked options versus vertical spreads
- +50%, -30%, and 15/21-trading-day exits
- Actual historical option quotes where available; modeled prices must be labeled clearly

## Unit tests completed

The detector code includes handcrafted canonical cases for:

- Five red candles followed by the first 5-minute candle making a new high
- A repeated flat top followed by the first breakout candle
- A flat-top breakout that fails and closes back below the level

All three tests pass.
