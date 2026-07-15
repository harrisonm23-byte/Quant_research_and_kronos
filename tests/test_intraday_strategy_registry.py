import sys
from pathlib import Path

import pandas as pd

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))

import intraday_strategy_registry as registry
import intraday_strategy_runner as runner
import options_intraday_overlay as options


def test_registry_has_two_virtual_exits_per_symbol_setup():
    assert registry.validate_registry()
    assert len(registry.STRATEGIES) == 12
    for symbol in ("QQQ", "TQQQ"):
        for setup in ("L1", "L2", "L3"):
            specs = [
                s for s in registry.STRATEGIES
                if s.symbol == symbol and s.setup == setup
            ]
            assert {s.exit_mechanic for s in specs} == {"fixed_24", "fixed_eod"}
            assert len({s.risk_cluster for s in specs}) == 1


def test_qqq_is_paper_and_tqqq_is_watch():
    assert all(s.status == "paper" for s in registry.STRATEGIES if s.symbol == "QQQ")
    assert all(s.status == "watch" for s in registry.STRATEGIES if s.symbol == "TQQQ")


def test_planned_exits_are_same_session_and_capped():
    morning = pd.Timestamp("2026-07-15 10:00", tz="America/New_York")
    late = pd.Timestamp("2026-07-15 15:00", tz="America/New_York")
    assert runner.strategy_exit_ts(morning, "fixed_24") == pd.Timestamp(
        "2026-07-15 12:00", tz="America/New_York"
    )
    assert runner.strategy_exit_ts(late, "fixed_24") == pd.Timestamp(
        "2026-07-15 15:55", tz="America/New_York"
    )
    assert runner.strategy_exit_ts(morning, "fixed_eod") == pd.Timestamp(
        "2026-07-15 15:55", tz="America/New_York"
    )


def test_watch_strategies_require_explicit_inclusion():
    hit = {
        "symbol": "TQQQ",
        "setup": "L1",
        "signal_ts": pd.Timestamp("2026-07-15 10:00", tz="America/New_York"),
        "signal_close": 75.0,
        "rsi": 30.0,
        "vwap_dist": -0.003,
    }
    assert runner.expand_strategies([hit], include_watch=False) == []
    expanded = runner.expand_strategies([hit], include_watch=True)
    assert len(expanded) == 2
    assert {x["exit_mechanic"] for x in expanded} == {"fixed_24", "fixed_eod"}


def test_option_overlay_prices_are_positive_and_spread_is_capped():
    call = registry.OVERLAYS["QQQ_ATM_CALL_2DTE"]
    spread = registry.OVERLAYS["QQQ_CALL_SPREAD_1PCT_2DTE"]
    call_price = options.price_overlay(call, spot=500, years=2 / 252, iv=0.25)
    spread_price = options.price_overlay(spread, spot=500, years=2 / 252, iv=0.25)
    assert call_price > 0
    assert 0 < spread_price < call_price
    assert spread_price < 500 * spread.width


def test_option_strikes_stay_fixed_after_entry():
    call = registry.OVERLAYS["QQQ_ATM_CALL_2DTE"]
    strikes = options.overlay_strikes(call, entry_spot=500)
    entry = options.price_overlay(
        call, spot=500, years=2 / 252, iv=0.25, strikes=strikes
    )
    exit_value = options.price_overlay(
        call, spot=505, years=1.5 / 252, iv=0.25, strikes=strikes
    )
    assert strikes[0] == 500
    assert exit_value > entry

