# Trading Research Ledger

Frozen baseline + running findings. All results: Alpaca SIP data, adjustment=split/all,
signals on completed bars, fills next open, slippage 0.02%/side unless noted.
Discipline: a finding counts only if it (1) replicates on QQQ AND SPY, (2) holds on both
time halves (2017-2021 / 2022-2026), (3) has a mechanism story.

## 1. FROZEN BASELINE GRID (pre quant-readings, 2026-07-05)

Daily strategies, window 2017-04-01 -> 2026-07-01:

```
Strategy    Sym  TF    Ann%   WR%    PF  MaxDD  AvgDD  MedDD  AvgTr  MedTr  Hold  #Tr
DoubleSeven SPY  1D     5.1  74.2  1.89  -11.8  -2.45  -1.13  0.544  1.137   7.8   89
DoubleSeven QQQ  1D     9.2  75.0  2.22  -13.6  -3.06  -1.27  0.872  1.279   7.3   96
IBS<.20/.70 SPY  1D     7.0  63.2  1.53  -16.2  -1.94  -0.66  0.238  0.349   2.6  277
IBS<.20/.70 QQQ  1D    14.6  66.3  1.84  -15.6  -2.16  -0.89  0.454  0.612   2.4  291
5DayLow-A   SPY  1D     3.5  62.6  1.46  -12.5  -2.49  -1.06  0.189  0.384   1.7  179
5DayLow-A   QQQ  1D     8.6  64.9  1.83  -11.5  -2.20  -1.07  0.390  0.494   1.6  205
TripleRSI   SPY  1D     3.4  75.0  2.75   -8.5  -2.60  -1.37  0.787  1.089   5.2   40
TT-A (Mon)  SPY  1D     2.4  55.9  1.40   -7.7  -1.89  -1.17  0.131  0.091   1.0  179
TT-C (Mon)  SPY  1D     2.6  71.2  1.69  -14.9  -2.33  -1.13  0.317  0.519   2.7   80
LowerBand-B QQQ  1D     4.9  72.3  1.85  -12.6  -2.80  -1.14  0.568  1.008   3.8   83
IBS+RSI21   SPY  1D     1.2  66.3  1.24  -16.3  -4.01  -2.20  0.161  0.501   1.9   83
RSI2-Mod-A  SPY  1D    -2.0  66.7  0.56  -31.8  -6.61  -1.67 -0.770  0.890   7.0   21
Buy&Hold    SPY  1D    15.0    --    --  -33.8  -1.68  -0.48     --     --    --   --
Buy&Hold    QQQ  1D    21.1    --    --  -35.0  -2.71  -0.82     --     --    --   --
TQQQ: IBS 39.4%/-41.6dd | 5DL 25.2%/-35.0 | D7 12.3%/-71.0 (leverage scales edge AND pain ~2.7x)
SQQQ: ALL strategies negative (PF 0.58-1.08). Long-inverse mean reversion structurally dead.
```

KEEPERS (paper-trading spec): IBS<0.20/0.70 QQQ (+TQQQ at 1/3 size), DoubleSeven QQQ,
5DayLow-A QQQ. No stops. No intraday entry confirmation. Optional dip-limit open*0.9975.

CONTEXT GRADE v1 (frozen): at signal, +1 each: volume<=1.2x20d, SMA20>SMA50;
-1 each: RSI14<35, 3+ red closes, range>1.5xATR14.
GREEN(+2): WR 71.6%/69.0% (QQQ/SPY). RED(<0): 38-50% WR, negative -> SKIP.
Time-split: GREEN 2017-21 = 75-78% WR (passes); 2022-26 = 61-63% (real but decayed).

## 2. VALIDATED PRINCIPLES

- Condition x context always beats condition alone ("intonation" model)
- Multi-TF agreement helps HOLDS; earliest unconfirmed signal has the SCALP juice
  (alignment = maturity; confirmation is always bought with retracement, ~linearly)
- Quiet weakness reverts (62% type); violent/high-volume weakness continues
- Trend-break signals END regimes, they do not start opposite ones (post-break drift ~0)
- Best intraday break marker: close above BB midline (58% no-new-low). RSI bullish
  divergence is an ANTI-signal at 5m (24.5%)
- Down-walk breaks end downtrends (46.6% cont.); up-walk breaks are pauses (52.6% cont.)
- Overnight drift: ~2/3 of index return accrues close->open (SPY +10.0%/4.8%,
  QQQ +13.4%/7.1% overnight/intraday ann). Overnight-holding longs ride it;
  intraday-only shorts fight it.
- Friday IBS signals: +11-12pts WR, both symbols, both halves (stable adder)
- SKEW 5d-rise: +3-5pts, both symbols, 3 independent appearances (weak stable adder)
- Conditional-open 2x2 (2026-07-15, both symbols, halves stable): prev-DOWN + gap-DOWN
  is the ONLY quadrant with intraday edge (57%/56% O->C up, +0.07/+0.10%). Prev-UP +
  gap-DOWN = NO edge (dip only buyable after prior weakness — context law again).
  Gap-fill ladder (P touch prev close): -0.2..0% gap = 88-89%; -0.5..-0.2% = 69-80%;
  -1..-0.5% = 36-47%; <-1% = 23-26% (big gaps: biggest O->C bounce +0.17/+0.36% but
  green close only 11-19%). Day shape on down-down: LOW early (45% first hour, med
  ~90min), HIGH late (med 2-3.5h, 28-34% last hour) -> buy first-hour weakness,
  sell afternoon strength.
- Tight stops (0.15-0.25%) destroy sub-10bp edges. Mean reversion wants immediacy
  (buy the open), not confirmation.
- Bollinger-band WIDTH is a MAGNITUDE dial, not a direction (2026-07-10 discovery mode,
  SMA9/lowerBB corridor, both symbols both halves): wide bands -> 60min MFE ~0.22%/0.29%
  (SPY/QQQ) vs ~0.12%/0.16% normal (~2x range); tight bands -> ~0.08% (dead). Dwell/coil
  time barely moves it. The move is SYMMETRIC (skew ~-3% in-state; the +11% "long-coil"
  cell was an OVERLAP artifact -> de-overlapped race = 50/50, halves disagree). Compression
  = a VOLATILITY signal (options/straddle lens or confluence FILTER for "when a move is
  loaded"), never a directional share trade. Corridor state itself is symmetric-to-bearish.
  Release from corridor breaks UP 63-64% but up-release is choppy; the rarer DOWN-break is
  the one that travels (drift -0.02%, bigger MFE). METHOD NOTE: kill-filter would've logged
  this "no edge, debunked"; discovery mode extracted a usable WHEN-fact instead.
- Volume-surge breakout from a TIGHT range (2026-07-10, both symbols): real but weak
  and ASYMMETRIC. Base tight-range break continues ~55%; surge (dvol>=2.5) lifts SPY to
  62% (QQQ 55%). BUT ~88% of breakouts never travel 0.3% in 60min (chop); drift ~0;
  moves tiny (favMFE ~0.08-0.12%). The edge is a DOWNSIDE edge: surge DOWN-break
  continues SPY 68% / QQQ 62%; surge UP-break FADES (QQQ 45%, mean-reverts). Consistent
  with index up-drift: downside momentum bursts carry, upside breaks get sold. Live
  candidate = surge downside break via PUT convexity (directional hit + defined risk),
  needs down-subset half-split + option-cost score before promotion.
  DRILLED (down-subset, n=419 SPY/502 QQQ): the flush is FAST not sustained -- hits -0.3%
  first on the movers, but P(still down at 60min)=42-43% (< coin flip) -> it BOUNCES.
  Put breakeven needs premium <6-11bp (hold) or <13-23bp (perfect-low exit); real 0DTE
  puts cost multiples -> NOT tradeable via options (move < premium). Verdict: discretionary
  SCALP tell only; the reverting half aligns with flush-low-holds-70% + daily MR book
  (fade the flush, don't chase). Options overlay stays on the DAILY high-WR signals.
- SMA9 crosses (5m) are highly PREDICTABLE but carry NO post-cross edge (2026-07-10,
  n=97k events/symbol, both symbols+halves). P(next bar closes across) spans 3% ->
  ~40% by state: close-to-line distance (>0.15% away = 3-5%; within 0.03% = 35-44%),
  counter-trend closes (0 = 7%; 2 = 32-36%), EMA9 already opposed (24-30%) vs leading
  (8-9%), RSI trendward, and QUIET volume crosses more than loud (loud = trend feeding,
  4th confirmation of the volume law). Best combo: EMA9-opposed + 2 counter closes =
  38-44% vs EMA9-leading + 0 counter = 3% (13x spread, stable halves). BUT fwd 30-60m
  after cross ~= 0bp vs baseline -> trend-break law holds at 5m: crosses END drifts,
  never start reversals. Use as exit-timing / chase-avoidance state, never as an entry.

## 3. DEBUNKED (tested, dead)

- Overbought short (VWAP+RSI+upper BB), all TFs -- worse than baseline intraday
- ORB 15-min breakout as marketed -- every config negative both symbols (best PF 0.99)
- SMA9-kiss short in band-walks (21.8% WR); first-break entries (~0 edge)
- VWAP "magnet" close (base rate illusion: 32% of all days close within 0.10%)
- MACD/RSI plateau "roundness" (indicator artifact, 51-53% everywhere)
- Intraday (5m-1h) versions of daily mean reversion (edge 2-4bp < costs)
- CBOE SKEW level for next-day timing (no pattern)
- Waiting for VWAP-reclaim before entering daily longs (-3 to -10bp)
- Turnaround-Tuesday on SQQQ / anything long SQQQ
- Warrior chart patterns, GENERIC LIQUID TRACK ONLY (2026-07-15, user's spec+detectors,
  SPY+QQQ 5m 2016-2026 + SPY 1m 2022-26, next-bar-open + their R-sim, vs matched
  baseline): ALL NEGATIVE-TO-NULL on index ETFs. 5-candle exhaustion reversal: loses
  BOTH sides both symbols (PF 0.71-0.77, races 44-47% vs 48-49% base) = confirmation
  tax + trend-break law (the "first new high" buys the retracement, starts nothing).
  Flat-top breakout (3-touch): PF 0.94-0.96, drift 0 = unconditioned breakouts dead
  without the volume filter (S/R chapter redux). Failed-flat-top bull-trap short:
  PF 0.42-0.47 (intrabar fill caveat noted; next-bar-open also negative). 10-candle
  1m reversal: n=254, faint long whisper (55% race, +1.4bp) < costs. NOT TESTED:
  track 2 (small-cap premarket gappers, the guide's native habitat) — requires
  dynamic gapper universe build; verdict explicitly does NOT extend there.
  TRACK 2 COMPLETED (2026-07-16): 5,384 gapper events (gap>=7%, $1.5-20, >=$5M traded,
  top-8/day, 2023-07->2026-07, 1m bars incl premarket, 30bps costs, next-bar-open).
  72,343 signals. Even in native habitat, mechanical patterns LOSE: failed-flat-top
  short -0.52%/tr (n=30k); rev5 5m both sides -0.3 to -0.8%; rev10 1m short -1.26%
  (never short gapper strength); flat-top long mean 0.0% but MEDIAN -0.48% (lottery
  structure: most fade, rare monsters carry the mean; medMFE 3.3%). ONE whisper:
  gap-and-go PMH break on MODERATE gappers (7-12%): +1.43%/tr n=504 — but year-
  unstable (+0.6/-0.0/+0.7/-1.6) and extreme gappers (>25%) FADE the break (-0.61%).
  CAVEATS: survivorship FLATTERS longs (still lose -> damning); 30bps generous for
  $3 stocks; halts/borrow unmodeled (shorts worse in practice). CONCLUSION: the
  guide's edge, if real, lives in discretionary selection + loss-cutting, not the
  mechanical entries. Chapter closed; nothing deploys.
- VWAP-rollover fade (EMA9 up->down turn while extended above a still-rising VWAP,
  target VWAP tag, 0.25% stop): structure REAL (morning rollovers tag VWAP 60-66%,
  both symbols) but ALL 24 cells net-negative — win capped at distance-to-VWAP (~0.12%)
  vs 0.25% stop + 4bp costs. Farther from VWAP = MORE continuation (fuel, not stretch).
  Kept as discretionary screen only, not a system.
- Early loud spike -> quiet counter-float -> afternoon snap-back in spike direction:
  NO (2026-07-10, 5m dvol deseasonalized by time-of-day). Pure form barely exists
  (~6 days/decade — spike days stay loud, 95% print another 2x bar). Relaxed form
  (n=182 SPY / 263 QQQ float-against days): afternoon goes spike-direction only
  46-49% (coin flip), halves flip sign = noise. The float direction mildly persists
  instead. USEFUL BYPRODUCT: after an early loud DOWN-spike that then floats up
  quietly, the morning flush low holds the rest of day ~70-73% (revisit only 27-30%)
  -> loud early flush + quiet recovery = the low is probably in (climax reading).

## 4. THE GOAL (operationalized)

Signal set: fires >=2-3x/week combined; per-signal WR 95% CI LOWER bound >=60%;
median favorable move >=0.3% within 1-2 sessions; survives both-symbol + time-split +
forward paper log; positive expectancy via defined option structure (put-spread family).

## 5. LOOP STATE (2026-07-05)

Candidates in gauntlet:
- Gap-up continuation: gap>+0.5% -> P(open->close up)=62.4% n=399 SPY; best cell
  gap>+0.5% + first-30m dip>0.2% -> rest-of-day +0.28%, 59% (n=83). NEEDS: QQQ + time-split.
- Grade v2 = v1 + Friday(+1) + SKEW-rising(+0.5): needs re-time-split.
- Corridor v2 (walk-confirm entry, target exits, context filters): queued.
- ToM + gap-fill stats: bugged first pass, re-run queued.

Live forward test: long QQQ from 725.71 (7/1 IBS signal), -1.79% @ 7/2 close, exit on
IBS>0.70 close. Next session 2026-07-06.

## 6. OPTIONS LAYER

Sim (BS, modeled IV, 2% costs): short put spread (ATM/-2%) robust across IV regimes
(+8-12% of max risk per trade, WR 66-70%) on TT-A and 5DayLow signals. Long calls
only viable in spike-exit mode; hold-to-expiry negative. High-WR small-move edges
feed premium SELLING. Needs real-chain validation in paper account.

## 7. COMMODITY TREND SLEEVE (gauntlet PASSED 2026-07-05 -> DEPLOYED KEEPER #4)

MA-cross long/flat on commodity ETFs, 10/40 daily (pre-registered pair; edge is
parameter-robust across 5/20 -> 20/100):
- USO: PF 2.29 both halves (1.85/1.96), +225% total vs B&H +39%. Ex-2020-episode
  +223% -> NOT a one-event fluke. Strongest case.
- GLD: PF 2.69 full; regime-loaded (flat 2016-21, +98% 2022-26). Keep with caveat.
- SLV: marginal (PF 1.90, recent-loaded). Watchlist only.
- UNG: FAILS everything (PF<0.8) -- contango decay too fast for long-only trend. Excluded.
Deploy spec: GLD + USO, 10/40 daily cross, long/flat, next-open fills. ~8 fires/yr,
multi-week holds, uncorrelated with equity mean-reversion set (portfolio diversifier,
not a frequency fix). PROMOTED to deployed keeper #4 at reduced sizing (n=38-43 trades
per asset vs 200-300 for equity keepers -> wider uncertainty, smaller allocation).
Killed post-goal: gap-up continuation (62% full-sample -> 51-54% recent half).
Grade v2 (v1 + Friday+1 + SKEWrising+0.5, A-tier >=2.5): recent-half WR 68-74%,
+0.65%/trade, ~2 fires/mo. Best signal to date.

## 8. SUPPORT/RESISTANCE ZONES (v1 1m + v2 5m, 2026-07-05)

v2 zones: clustered pivots, VALID after >=2 touches >=60min apart, ATR-scaled width,
close-confirmed breaks, retest tracking. SPY 5m 2016-2026, 9,168 events.
- TOUCH-COUNT FOLKLORE INVERTED: more touches = weaker level, monotone both sides
  (support 45.2->43.3->41.5% by touch#; resistance 50.5->48.6->46.1%)
- Support tests: NULL even with proper zones (44.4% vs 45.0% base). At 1m: below chance.
- Resistance test x LOUD volume (>1.5x): holds only 43.0% -> P(break up)=57.0% vs 45.0%
  base = +12pts, largest intraday cell ever measured; EV ~3.5bp vs 4bp costs = breakeven.
  Quiet-vol arrival at resistance: holds 52.6%. Volume law's 4th independent domain.
- Break->retest->go: DEBUNKED (flip holds 52.5% vs 55.0% base)
- Support breaks close-confirmed do NOT out-continue baseline (confirmation eats the move)
- Rulebook context adopted: fresh zones > tested zones; never fade loud volume into a
  level; retests are not confirmation.

## 9. CRISIS SLEEVE E + VIX BUY-THE-FEAR (2026-07-06)

VIX first-cross forward returns (SPY/QQQ, 2016-2026, ~5 independent crisis regimes):
  VIX>35: SPY +5.6/11.4/19.3% (21/63/126d), 92% WR; QQQ +7.4/14.7/25.0%
  VIX>40: +100% WR at 63d+ but n=5. Buy-the-fear = mean reversion at macro scale (6th domain)
Leverage/inverse at VIX>35 (fwd 63d, worst drawdown-during-hold):
  QQQ  +14% (worst -20%)  |  TQQQ +41% (worst -58%)  |  SQQQ -38% 0% WR (worst -62%)
TQQQ oversold-entry-then-hold (RSI14<30 / 50d-low): +27-68% fwd, 75-87% WR, BUT
  worst intra-hold drawdown -57 to -78%. "Hold-to-today" numbers (+800-3500%) are PURE
  survivorship (QQQ 10x'd 2016-26; TQQQ launched 2010, never saw a real bear).
  DotCom -83% Nasdaq => 3x = -99%+ permanent. "Just hold TQQQ" dies in a structural bear.

SLEEVE E — Crisis buy-the-fear (dormant ~95%, fires ~1x/1-2yr on VIX>35 first cross):
  CORE:       QQQ shares or ~12mo LEAPS (+14-25%/3-6mo, -20% worst). The default.
  AGGRESSIVE: TQQQ SMALL size + EXIT on recovery (never hold-forever). +41% but -58% tail.
  NEVER:      SQQQ (0% WR, structural anti-trade). Size for the drawdown not the average.
  Fits book's 30% cash sleeve = the dry powder this trigger is for. Wins whether the
  crash is this year or after a 2-more-year melt-up; structure doesn't need the timing.

IV-RANK GATE (phase 2, low priority): gate put-spread overlay sells on QQQ IV Rank>=50
  & IV>HV. Dormant 6-12mo until forward IV history fills. Marginal, logged not prioritized.

EXTERNAL VALIDATION: a public 9,120-backtest / 30-asset / 2010-2025 survivor terminal
  ranked top survivors as RSI/Keltner/Zscore Revert (mean reversion) + Turtle/ADX/Dual
  Momentum (trend), best Sharpes 1.0-1.18, scored by cross-asset survival count. Same two
  families, same replication discipline, same realistic Sharpe range we converged on.
