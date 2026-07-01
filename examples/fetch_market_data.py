"""
fetch_market_data.py - Download OHLCV history for US tickers/ETFs into the CSV
format that analyze_market.py expects.

Daily data is keyless (stockanalysis.com / Nasdaq fallback). Intraday data
(1m/5m/15m/30m/1h) uses the Alpaca Market Data API — pass your credentials
via --alpaca-key / --alpaca-secret or the ALPACA_API_KEY / ALPACA_API_SECRET
environment variables. Paper-trading keys work fine.

Output columns are:

    timestamps,open,high,low,close,volume,amount

sorted oldest-first. `amount` (turnover) is approximated as close*volume when
the source doesn't provide it.

Examples
--------
  # Five years of daily bars (no key needed)
  python fetch_market_data.py SPY QQQ --range 5Y --outdir data

  # Intraday via Alpaca
  python fetch_market_data.py SPY QQQ --interval 1h --days 90 --outdir data
  python fetch_market_data.py SPY QQQ --interval 5m --days 30 --outdir data

  # Then forecast / backtest with the sibling tool:
  python analyze_market.py backtest --csv data/SPY_1h.csv --model-size small --pred-len 10
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

RANGE_ALIASES = {"1M": "1M", "6M": "6M", "1Y": "1Y", "2Y": "3Y", "3Y": "3Y",
                 "5Y": "5Y", "10Y": "10Y", "MAX": "10Y", "ALL": "10Y"}

ALPACA_INTERVALS = {
    "1m": "1Min", "1min": "1Min",
    "5m": "5Min", "5min": "5Min",
    "15m": "15Min", "15min": "15Min",
    "30m": "30Min", "30min": "30Min",
    "60m": "1Hour", "1h": "1Hour", "1hour": "1Hour", "hourly": "1Hour",
}


def _get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


# --------------------------------------------------------------------------- #
# Daily sources (keyless)
# --------------------------------------------------------------------------- #
def fetch_stockanalysis(symbol, rng):
    token = RANGE_ALIASES.get(rng.upper(), rng)
    url = (f"https://stockanalysis.com/api/symbol/e/{symbol.upper()}"
           f"/history?range={token}&period=Daily")
    payload = json.loads(_get(url))
    if payload.get("status") != 200 or "data" not in payload:
        raise ValueError(f"stockanalysis returned status={payload.get('status')}")
    rows = []
    for d in payload["data"]:
        vol = float(d.get("v") or 0.0)
        close = float(d["c"])
        rows.append({
            "timestamps": d["t"],
            "open": float(d["o"]), "high": float(d["h"]),
            "low": float(d["l"]), "close": close,
            "volume": vol, "amount": round(close * vol, 2),
        })
    rows.sort(key=lambda r: r["timestamps"])
    return rows


def fetch_nasdaq(symbol, rng):
    url = (f"https://api.nasdaq.com/api/quote/{symbol.upper()}/historical"
           f"?assetclass=etf&fromdate=2016-01-01&todate=2099-01-01&limit=99999")
    payload = json.loads(_get(url, headers={"User-Agent": UA, "Accept": "application/json"}))
    table = (payload.get("data") or {}).get("tradesTable") or {}
    rows = []
    for d in table.get("rows", []):
        def num(x):
            return float(str(x).replace("$", "").replace(",", "")) if x not in (None, "") else 0.0
        mm, dd, yy = d["date"].split("/")
        close = num(d.get("close"))
        vol = num(d.get("volume"))
        rows.append({
            "timestamps": f"{yy}-{mm}-{dd}",
            "open": num(d.get("open")), "high": num(d.get("high")),
            "low": num(d.get("low")), "close": close,
            "volume": vol, "amount": round(close * vol, 2),
        })
    rows.sort(key=lambda r: r["timestamps"])
    return rows


# --------------------------------------------------------------------------- #
# Intraday via Alpaca
# --------------------------------------------------------------------------- #
def fetch_alpaca(symbol, timeframe, api_key, api_secret, days=60):
    """Paginated intraday OHLCV from the Alpaca Market Data v2 API.

    `timeframe` is an Alpaca token (1Min/5Min/15Min/30Min/1Hour). Fetches the
    last `days` calendar days, paginating via next_page_token (Alpaca caps
    responses at 10 000 bars).
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    base = (f"https://data.alpaca.markets/v2/stocks/{symbol.upper()}/bars"
            f"?timeframe={timeframe}"
            f"&start={start.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&end={end.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            f"&limit=10000&feed=iex&adjustment=raw")
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": api_secret,
    }
    rows = []
    url = base
    page = 0
    while url:
        payload = json.loads(_get(url, headers=headers))
        for b in payload.get("bars") or []:
            close = float(b["c"]); vol = float(b["v"])
            rows.append({
                "timestamps": b["t"].replace("T", " ").replace("Z", ""),
                "open": float(b["o"]), "high": float(b["h"]),
                "low": float(b["l"]), "close": close,
                "volume": vol, "amount": round(close * vol, 2),
            })
        npt = payload.get("next_page_token")
        if npt:
            page += 1
            url = base + f"&page_token={npt}"
            time.sleep(0.3)
        else:
            url = None
    rows.sort(key=lambda r: r["timestamps"])
    return rows


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def fetch(symbol, rng, interval="daily", alpaca_key=None, alpaca_secret=None, days=60):
    if interval != "daily":
        tf = ALPACA_INTERVALS.get(interval.lower())
        if not tf:
            raise ValueError(f"Unsupported interval '{interval}'. "
                             f"Use daily or one of: {sorted(set(ALPACA_INTERVALS))}")
        if not alpaca_key or not alpaca_secret:
            raise RuntimeError(
                "Intraday data needs Alpaca API credentials. Pass --alpaca-key "
                "and --alpaca-secret (or set ALPACA_API_KEY / ALPACA_API_SECRET).")
        rows = fetch_alpaca(symbol, tf, alpaca_key, alpaca_secret, days=days)
        if rows:
            print(f"  {symbol}: {len(rows)} {tf} bars via alpaca "
                  f"({rows[0]['timestamps']} -> {rows[-1]['timestamps']})")
            return rows
        raise RuntimeError(f"Alpaca returned no bars for {symbol} {tf}")

    errors = []
    for name, fn in (("stockanalysis", fetch_stockanalysis), ("nasdaq", fetch_nasdaq)):
        try:
            rows = fn(symbol, rng)
            if rows:
                print(f"  {symbol}: {len(rows)} bars via {name} "
                      f"({rows[0]['timestamps']} -> {rows[-1]['timestamps']})")
                return rows
            errors.append(f"{name}: empty")
        except Exception as e:  # noqa: BLE001 - report and try next source
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"all sources failed for {symbol}: {'; '.join(errors)}")


def write_csv(rows, path):
    cols = ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def main():
    p = argparse.ArgumentParser(description="Fetch OHLCV CSVs for analyze_market.py.")
    p.add_argument("symbols", nargs="+", help="Tickers, e.g. SPY QQQ AAPL")
    p.add_argument("--range", default="5Y", help="Daily history window: 1M/6M/1Y/3Y/5Y/10Y (default 5Y).")
    p.add_argument("--interval", default="daily",
                   help="Bar size: daily (keyless) or intraday 1m/5m/15m/30m/1h (Alpaca).")
    p.add_argument("--alpaca-key", default=os.environ.get("ALPACA_API_KEY"),
                   help="Alpaca API key ID (or set ALPACA_API_KEY env var).")
    p.add_argument("--alpaca-secret", default=os.environ.get("ALPACA_API_SECRET"),
                   help="Alpaca API secret key (or set ALPACA_API_SECRET env var).")
    p.add_argument("--days", type=int, default=60,
                   help="Calendar days of intraday history to fetch (default 60).")
    p.add_argument("--outdir", default="data", help="Directory to write <SYMBOL>.csv into.")
    args = p.parse_args()

    suffix = "" if args.interval == "daily" else f"_{args.interval.lower()}"
    print(f"Fetching {len(args.symbols)} symbol(s), interval={args.interval}")
    failures = []
    for sym in args.symbols:
        try:
            rows = fetch(sym, args.range, interval=args.interval,
                         alpaca_key=args.alpaca_key, alpaca_secret=args.alpaca_secret,
                         days=args.days)
            out = os.path.join(args.outdir, f"{sym.upper()}{suffix}.csv")
            write_csv(rows, out)
            print(f"  -> wrote {out}")
        except Exception as e:  # noqa: BLE001
            print(f"  !! {sym} failed: {e}", file=sys.stderr)
            failures.append(sym)
    if failures:
        sys.exit(f"Failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
