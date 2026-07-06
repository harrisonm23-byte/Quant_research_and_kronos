"""Running paper-trading returns tracker.

Re-run any time. Reads book_state.json (positions + closed trades), marks open
positions to latest Alpaca daily close, prints per-sleeve and total running
returns. Update book_state.json as the bot opens/closes trades (or by hand).

Env: ALPACA_API_KEY / ALPACA_API_SECRET (paper). No keys are stored in this file.
"""
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "book_state.json")
KEY = os.environ.get("ALPACA_API_KEY", "")
SEC = os.environ.get("ALPACA_API_SECRET", "")


def latest_close(sym):
    end = (datetime.now(timezone.utc) - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
    is_crypto = "/" in sym
    if is_crypto:
        p = {"symbols": sym, "timeframe": "1Day", "start": "2026-06-01", "end": end, "limit": "50"}
        url = "https://data.alpaca.markets/v1beta3/crypto/us/bars?" + urllib.parse.urlencode(p)
    else:
        p = {"timeframe": "1Day", "start": "2026-06-01", "end": end, "limit": "50",
             "feed": "sip", "adjustment": "all"}
        url = f"https://data.alpaca.markets/v2/stocks/{sym}/bars?" + urllib.parse.urlencode(p)
    r = urllib.request.Request(url, headers={"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC})
    with urllib.request.urlopen(r, timeout=30) as resp:
        data = json.loads(resp.read())
    bars = data["bars"][sym] if is_crypto else data.get("bars", [])
    return bars[-1]["c"] if bars else None


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"sleeves": {}, "open": [], "closed": [], "start_date": "2026-07-02"}


def main():
    st = load_state()
    print(f"=== PAPER BOOK — running returns (as of {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC) ===")
    print(f"Tracking since {st.get('start_date','?')}\n")

    by_sleeve = {}
    # closed trades
    for t in st.get("closed", []):
        by_sleeve.setdefault(t["sleeve"], []).append(t["ret"])
    # open positions marked to market
    print(f"{'OPEN POSITIONS':<28s}{'entry':>9s}{'mark':>9s}{'unreal%':>9s}")
    for pos in st.get("open", []):
        mk = latest_close(pos["symbol"])
        if mk is None:
            print(f"  {pos['strategy']:<26s} (no data)")
            continue
        ur = mk / pos["entry_px"] - 1
        by_sleeve.setdefault(pos["sleeve"], []).append(ur)
        print(f"  {pos['strategy']+' '+pos['symbol']:<26s}{pos['entry_px']:>9.2f}{mk:>9.2f}{ur*100:>+8.2f}%")

    print(f"\n{'SLEEVE':<10s}{'trades':>8s}{'wins':>6s}{'WR':>7s}{'cum ret%':>10s}")
    total_ret = 1.0
    weights = {"A": 0.40, "B": 0.20, "C": 0.10, "D": 0.30}
    book = 0.0
    for sl in ["A", "B", "C", "D"]:
        rets = by_sleeve.get(sl, [])
        if not rets:
            print(f"  {sl:<8s}{'0':>8s}{'--':>6s}{'--':>7s}{'0.00':>10s}")
            continue
        import numpy as np
        a = np.array(rets)
        cum = np.prod(1 + a) - 1
        wr = (a > 0).mean()
        book += weights[sl] * cum
        print(f"  {sl:<8s}{len(a):>8d}{int((a>0).sum()):>6d}{wr:>7.1%}{cum*100:>+10.2f}")
    print(f"\n  BOOK (weighted, {int(sum(weights.values())*100)}% deployed): {book*100:+.2f}%")
    print("  (paper phase: judge on 30+ closed trades, not the first weeks)")


if __name__ == "__main__":
    main()
