#!/usr/bin/env python3
"""Operational registry for first-signal intraday BB-fade candidates.

These are forward-paper candidates, not production-approved systems.  QQQ has
an independent 146-session historical replication; TQQQ has only the recent
~60-session sample and therefore remains WATCH.  Each setup is evaluated with
two virtual exits (120m and EOD), but variants share one risk cluster and must
not be stacked as separate live positions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OverlaySpec:
    overlay_id: str
    underlying: str
    structure: str
    dte: int
    moneyness: float
    width: float | None
    premium_cap_usd: int
    status: str
    notes: str


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    symbol: str
    setup: str
    setup_mask: str
    exit_mechanic: str
    status: str
    risk_cluster: str
    overlay_ids: tuple[str, ...]
    evidence: str
    notes: str


OVERLAYS = {
    # Two DTE avoids same-day expiry mechanics while preserving intraday gamma.
    # These remain modeled/paper structures until real-chain fills are logged.
    "QQQ_ATM_CALL_2DTE": OverlaySpec(
        overlay_id="QQQ_ATM_CALL_2DTE",
        underlying="QQQ",
        structure="long_call",
        dte=2,
        moneyness=0.00,
        width=None,
        premium_cap_usd=250,
        status="paper_model",
        notes="ATM call; close with underlying strategy; no averaging.",
    ),
    "QQQ_CALL_SPREAD_1PCT_2DTE": OverlaySpec(
        overlay_id="QQQ_CALL_SPREAD_1PCT_2DTE",
        underlying="QQQ",
        structure="bull_call_spread",
        dte=2,
        moneyness=0.00,
        width=0.01,
        premium_cap_usd=250,
        status="paper_model",
        notes="Long ATM call / short 1% OTM call; defined debit risk.",
    ),
    "TQQQ_ATM_CALL_2DTE": OverlaySpec(
        overlay_id="TQQQ_ATM_CALL_2DTE",
        underlying="TQQQ",
        structure="long_call",
        dte=2,
        moneyness=0.00,
        width=None,
        premium_cap_usd=100,
        status="research_only",
        notes="Double leverage; do not deploy before real-chain spread/liquidity review.",
    ),
}


def _strategies():
    specs = []
    for symbol in ("QQQ", "TQQQ"):
        status = "paper" if symbol == "QQQ" else "watch"
        overlays = (
            ("QQQ_ATM_CALL_2DTE", "QQQ_CALL_SPREAD_1PCT_2DTE")
            if symbol == "QQQ"
            else ("TQQQ_ATM_CALL_2DTE",)
        )
        evidence = (
            "QQQ: 146 historical sessions + recent 60-session replication"
            if symbol == "QQQ"
            else "TQQQ: recent ~60 sessions only; long-history validation pending"
        )
        for setup in ("L1", "L2", "L3"):
            for mechanic, suffix in (("fixed_24", "120M"), ("fixed_eod", "EOD")):
                specs.append(StrategySpec(
                    strategy_id=f"{symbol}_{setup}_FIRST_{suffix}",
                    symbol=symbol,
                    setup=setup,
                    setup_mask=f"{setup}_first",
                    exit_mechanic=mechanic,
                    status=status,
                    risk_cluster=f"{symbol}_{setup}_FIRST",
                    overlay_ids=overlays,
                    evidence=evidence,
                    notes=(
                        "Enter next 5m open; first setup signal/session; same-session exit. "
                        "120M and EOD are virtual alternatives sharing one capital allocation."
                    ),
                ))
    return tuple(specs)


STRATEGIES = _strategies()
BY_ID = {s.strategy_id: s for s in STRATEGIES}


def rows():
    """Flat rows for CLI display or CSV export."""
    return [asdict(s) for s in STRATEGIES]


def validate_registry():
    ids = [s.strategy_id for s in STRATEGIES]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate strategy_id")
    unknown = {
        overlay_id
        for spec in STRATEGIES
        for overlay_id in spec.overlay_ids
        if overlay_id not in OVERLAYS
    }
    if unknown:
        raise ValueError(f"unknown overlays: {sorted(unknown)}")
    for spec in STRATEGIES:
        if spec.exit_mechanic not in {"fixed_24", "fixed_eod"}:
            raise ValueError(f"unsupported exit: {spec.exit_mechanic}")
        if not spec.setup_mask.endswith("_first"):
            raise ValueError(f"not first-per-day: {spec.strategy_id}")
    return True


validate_registry()

