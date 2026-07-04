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
- Tight stops (0.15-0.25%) destroy sub-10bp edges. Mean reversion wants immediacy
  (buy the open), not confirmation.

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
