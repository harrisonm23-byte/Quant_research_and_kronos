"""
fetch_market_data.py - Download OHLCV history for US tickers/ETFs into the CSV
format that analyze_market.py expects (no API key required).

Primary source is stockanalysis.com's public JSON endpoint; if that fails the
script falls back to Nasdaq's public API. Output columns are:

    timestamps,open,high,low,close,volume,amount

sorted oldest-first. `amount` (turnover) is approximated as close*volume when the
source doesn't provide it, which is what Kronos expects for the amount channel.

Examples
--------
  # Five years of daily bars for two ETFs into ./data/
  python fetch_market_data.py SPY QQQ --range 5Y --outdir data

  # Then forecast / backtest with the sibling tool:
  python analyze_market.py backtest --csv data/SPY.csv --model-size small --pred-len 10
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# stockanalysis.com accepts these range tokens; we map a few friendly aliases.
RANGE_ALIASES = {"1M": "1M", "6M": "6M", "1Y": "1Y", "2Y": "3Y", "3Y": "3Y",
                 "5Y": "5Y", "10Y": "10Y", "MAX": "10Y", "ALL": "10Y"}

# Friendly interval aliases -> Alpha Vantage interval tokens. "daily" is handled
# separately by the keyless sources above.
INTERVAL_ALIASES = {
    "1m": "1min", "1min": "1min",
    "5m": "5min", "5min": "5min",
    "15m": "15min", "15min": "15min",
    "30m": "30min", "30min": "30min",
    "60m": "60min", "1h": "60min", "60min": "60min", "hourly": "60min",
}


def _get(url, headers=None, timeout=30):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def fetch_stockanalysis(symbol, rng):
    """Return list of dict rows (oldest-first) from stockanalysis.com, or raise."""
    token = RANGE_ALIASES.get(rng.upper(), rng)
    url = (f"https://stockanalysis.com/api/symbol/e/{symbol.upper()}"
           f"/history?range={token}&period=Daily")
    payload = json.loads(_get(url))
    if payload.get("status") != 200 or "data" not in payload:
        raise ValueError(f"stockanalysis returned status={payload.get('status')}")
    rows = []
    for d in payload["data"]:
        # t=date o=open h=high l=low c=close v=volume a=adjClose
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
    """Fallback: Nasdaq public API. Approximates date range with a wide window."""
    # Nasdaq wants explicit dates; ask for a generous window and let downstream
    # tools slice with --lookback.
    years = {"1M": 1, "6M": 1, "1Y": 2, "3Y": 4, "5Y": 6, "10Y": 11}.get(rng.upper(), 6)
    url = (f"https://api.nasdaq.com/api/quote/{symbol.upper()}/historical"
           f"?assetclass=etf&fromdate=2016-01-01&todate=2099-01-01&limit=99999")
    payload = json.loads(_get(url, headers={"User-Agent": UA, "Accept": "application/json"}))
    table = (payload.get("data") or {}).get("tradesTable") or {}
    rows = []
    for d in table.get("rows", []):
        def num(x):
            return float(str(x).replace("$", "").replace(",", "")) if x not in (None, "") else 0.0
        # date format MM/DD/YYYY -> YYYY-MM-DD
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


def fetch_alphavantage(symbol, interval, api_key, months=None):
    """Intraday OHLCV from Alpha Vantage. Requires a free API key.

    `interval` is an Alpha Vantage token (1min/5min/15min/30min/60min). If
    `months` (list of 'YYYY-MM') is given, fetch each historical month and
    concatenate; otherwise fetch the most recent ~1-2 months (outputsize=full).
    """
    base = ("https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
            f"&symbol={symbol.upper()}&interval={interval}&outputsize=full"
            f"&adjusted=false&extended_hours=false&apikey={api_key}")
    targets = [base + f"&month={m}" for m in months] if months else [base]
    rows, key = [], f"Time Series ({interval})"
    for url in targets:
        payload = json.loads(_get(url))
        if key not in payload:
            # AV returns Note/Information on rate-limit or bad key/symbol.
            msg = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or str(payload)[:120]
            raise ValueError(f"Alpha Vantage: {msg}")
        for ts, d in payload[key].items():
            close = float(d["4. close"]); vol = float(d["5. volume"])
            rows.append({
                "timestamps": ts,
                "open": float(d["1. open"]), "high": float(d["2. high"]),
                "low": float(d["3. low"]), "close": close,
                "volume": vol, "amount": round(close * vol, 2),
            })
    rows.sort(key=lambda r: r["timestamps"])
    return rows


def fetch(symbol, rng, interval="daily", av_key=None, months=None):
    # Intraday => Alpha Vantage only (keyless sources are daily-only).
    if interval != "daily":
        av_interval = INTERVAL_ALIASES.get(interval.lower())
        if not av_interval:
            raise ValueError(f"Unsupported interval '{interval}'. "
                             f"Use daily or one of: {sorted(set(INTERVAL_ALIASES))}")
        if not av_key:
            raise RuntimeError("Intraday data needs an Alpha Vantage API key "
                               "(--av-key or ALPHAVANTAGE_API_KEY env). Free at "
                               "https://www.alphavantage.co/support/#api-key")
        rows = fetch_alphavantage(symbol, av_interval, av_key, months)
        if rows:
            print(f"  {symbol}: {len(rows)} {av_interval} bars via alphavantage "
                  f"({rows[0]['timestamps']} -> {rows[-1]['timestamps']})")
            return rows
        raise RuntimeError(f"Alpha Vantage returned no rows for {symbol} {av_interval}")

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
    p = argparse.ArgumentParser(description="Fetch OHLCV CSVs for analyze_market.py (no API key).")
    p.add_argument("symbols", nargs="+", help="Tickers, e.g. SPY QQQ AAPL")
    p.add_argument("--range", default="5Y", help="Daily history window: 1M/6M/1Y/3Y/5Y/10Y (default 5Y).")
    p.add_argument("--interval", default="daily",
                   help="Bar size: daily (keyless) or intraday 1m/5m/15m/30m/1h (needs Alpha Vantage key).")
    p.add_argument("--av-key", default=os.environ.get("ALPHAVANTAGE_API_KEY"),
                   help="Alpha Vantage API key for intraday (or set ALPHAVANTAGE_API_KEY).")
    p.add_argument("--months", default=None,
                   help="Comma list of YYYY-MM for historical intraday, e.g. '2026-04,2026-05,2026-06'.")
    p.add_argument("--outdir", default="data", help="Directory to write <SYMBOL>.csv into.")
    args = p.parse_args()

    months = [m.strip() for m in args.months.split(",")] if args.months else None
    suffix = "" if args.interval == "daily" else f"_{args.interval.lower()}"
    print(f"Fetching {len(args.symbols)} symbol(s), interval={args.interval}, range={args.range}")
    failures = []
    for sym in args.symbols:
        try:
            rows = fetch(sym, args.range, interval=args.interval, av_key=args.av_key, months=months)
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
