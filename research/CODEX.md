# THE CODEX — distilled rulebook (v1, 2026-07-11)

FINDINGS.md is the lab notebook. This is the rulebook: what we know, what we trade,
and how the pieces compose. Everything here survived both symbols + both time halves
unless marked otherwise.

## I. THE LAWS (cross-confirmed market structure)

L1. TIMEFRAME LAW — Edges scale with timeframe. Daily signals earn 15-90bp/trade and
    survive costs; intraday (1m-1h) versions of the same structure earn 2-10bp and die
    at ~4bp costs. The market is efficient in proportion to how fast you trade it.
L2. VOLUME LAW (6 domains) — Loud volume = initiative (moves continue while fed).
    Quiet = coasting (drifts stall/revert). Quiet weakness reverts; violent weakness
    continues. Loud arrival at resistance breaks it; quiet arrival holds.
L3. MAGNITUDE =/= DIRECTION — Compression (tight BB) says nothing is coming; expansion
    (wide BB) says a ~2x move is loaded but NOT which way. Volatility is predictable;
    direction mostly is not. Trade WHEN-signals with vol structures, WHICH-WAY signals
    with direction.
L4. TREND-BREAK LAW — Breaks/crosses END regimes; they do not start opposite ones.
    Post-break drift ~ 0 at every TF tested. Never trade a break as a reversal entry.
L5. CONFIRMATION TAX — Confirmation is always bought with retracement (~linearly).
    Mean reversion wants immediacy; waiting for reclaim/confirm costs 3-10bp.
L6. OVERNIGHT DRIFT — ~2/3 of index return accrues close->open. Overnight longs ride
    a tailwind; intraday-only shorts fight it. (Why down-edges intraday stay small.)
L7. ASYMMETRY OF BREAKS — Index up-breaks fade (get sold); down-breaks travel briefly
    then bounce. The fast flush is real but reverts: flush low holds rest-of-day ~70%.
L8. GEOMETRY > WIN RATE — WR without payoff geometry is decoration (RSI2-Mod: 66.7% WR,
    PF 0.56). Score every idea on the full distribution, not the hit rate.
L9. LEVERAGE LAW — Leverage scales edge AND pain ~2.7x (TQQQ). Long-inverse (SQQQ)
    is structurally dead. Crisis buying wants shares/LEAPS, never hold-forever 3x.
L10. CONTEXT LAW — Condition x context beats condition alone, always. Single-variable
    tests understate real confluence; test the JOINT state.

## II. INTRADAY TELLS (real structure, not standalone trades — use as state/context)

T1. SMA9-cross countdown (5m): P(next-bar cross) spans 3% -> 40% by state:
    distance to line + counter-trend closes + EMA9 lead/lag + RSI + quiet volume.
    EMA9-opposed + 2 counter closes = endgame (38-44%); EMA9-leading + 0 counter = safe (3%).
    USE: exit timing / chase avoidance. Never an entry (L4).
T2. Compression gauge (5m): BB-width tercile = move-size forecast (L3).
    Tight = dead tape (MFE ~0.08%); wide = loaded (~0.22-0.29%). USE: when-filter.
T3. Flush-and-hold (5m): early loud down-spike then quiet upward float = low is in
    ~70-73%. No afternoon snap-back (46-49% = coin flip). USE: fade flushes, don't chase.
T4. Volume-surge downside break of tight range: fast flush, hits -0.3% first 62-68%,
    but reverts by 60min (P(down at exit) 42%). Too small for option premium.
    USE: scalp tell / entry-timing for daily MR signals (buy the flush into an IBS day).

## III. THE DEPLOYED BOOK (daily signals — where the edge actually pays)

WEIGHTS: AGG-MAX locked 2026-07-13 (user call: maximize, paper phase).
  A 55% / B 10% / C+ 20% / D 15%. Backtest ~25%/yr, realistic 15-19%, worst DD -35-45%.
  (Rejected: 100% A-TQQQ ~37% backtest -- single-family fragility, -67% DD, no powder.)
Sleeve A (55%) — Equity MR pool via TQQQ at pool signals (QQQ signals, TQQQ execution,
  1/3 size rule retired under AGG-MAX -- full sleeve, sized BY the -67% DD math), one
  position at a time, priority: IBS<0.20 (graded) > DoubleSeven > 5DayLow-A.
  Exit IBS>0.70 / D7 rules. No stops. Next-open fills. Grade v2 (vol<=1.2x,
  SMA20>SMA50, Friday +1, SKEW +0.5; RSI<35, 3+ red, wide range -1): skip <0.
Sleeve B (10%) — Commodity trend: GLD + USO 10/40 daily cross, long/flat. ~8 fires/yr.
Sleeve C+ (20%) — Crypto down-spike H2: vol>=2x 30d mean AND red day on
  BTC/ETH/DOGE/LINK -> long at signal close (paper: next practical fill), hold 2 days,
  no stops, 25% of sleeve per coin, ~60-65 fires/yr. Final bt: ann +19.5%, maxDD -17%,
  Sharpe 0.92, every K/H cell positive. DECAY FLAG: 2024-26 +0.58%/tr vs 2021-23
  +2.05%; 2026 YTD negative. Review at 25 forward fires. (BTC D7 retired into this.)
Sleeve D (15%) — Cash. Dry powder for Sleeve E.
Sleeve E (event) — VIX>35 first cross: buy QQQ shares/LEAPS (+14-25%/3-6mo, 92% WR,
  ~5 independent crises). TQQQ small + exit-on-recovery only. Never SQQQ.
Options overlay (build phase): short put spreads on A-tier + high-WR daily signals.
  High-WR small-move edges SELL premium; they never buy it (move < premium, T4 proof).

PENDING DECISION: conservative weights above vs aggressive (A-via-TQQQ 45/B 15/C 15/D 25).

## IV. SLEEVE C+ CANDIDATE — crypto volume-spike (user's independent research)

Source: harrisonm23-byte/Crypto_Data_Project. Daily, 9 coins, 2019-2026, yfinance.
  Down-spike (vol>=2x 30d, red day) -> hold 2d: excess +3.06%, t=4.67, stable OOS.
  Up-spike (green day) -> hold 10d: excess +6.11%, t=3.46, stable OOS.
  Combined Sharpe 1.65, positive every year, beta~0 to BTC.
Maps to our laws: down-spike H2 = L2 flush-reverts (daily scale); up-spike H10 = L2
loud-continues. Daily TF = L1 compliant. Two-signal structure = the same MR+trend
pair every survivor system converges to.
REPLICATION (2026-07-11, Alpaca data 2021+, independent source):
  CORE (their coins on Alpaca: BTC/ETH/DOGE/LINK):
    down-spike H2: excess +1.15%, t=3.05  -> REPLICATES
    up-spike H10:  excess +2.77%, t=2.25  -> replicates (weaker)
    de-overlapped trades net 20bps: down n=350 mean +1.21% WR 54%; up n=221 mean
    +3.52% but MEDIAN -0.04% WR 50% (all skew -- a few big winners carry it)
  OUT-OF-UNIVERSE (6 Alpaca coins NOT in their study: AVAX/SHIB/LTC/BCH/UNI/AAVE):
    down-spike: +0.44% excess, t=1.39 (weakly positive, WR 51%)
    up-spike:   -0.20% excess, t=-0.29 (DEAD)
VERDICT: down-spike H2 is the robust half (significant on 2 data sources, 2 windows,
  never negative anywhere -- textbook L2 flush-reverts). Up-spike H10 is real on the
  studied universe but universe-fragile + median~0: watchlist, not deploy.
DEPLOY (Sleeve C+): down-spike H2 on BTC+ETH+DOGE+LINK, ~60-65 fires/yr (~1.2/wk),
  next-day-open fills, no stops, hold 2 days. Up-spike H10: paper-watch only.

## V. METHOD (how we find things now)

1. DISCOVERY MODE first: map the response surface (MFE/MAE distributions, winners
   uncapped, all knobs swept). No kill-verdicts from single operationalizations.
2. De-overlap before believing any cell (overlapping bars fake skew).
3. THEN the gauntlet, only for deploy candidates: both symbols, both halves,
   n>=50, PF>=1.5 full / >=1.2 halves, beats costs on ITS payoff structure
   (shares linear vs options convex — score on how it's actually monetized).
4. Every kill must state WHICH encoding died, not that the pattern is false.
5. Paper-log forward, judge on 30+ closed trades.
