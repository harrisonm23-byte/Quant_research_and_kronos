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


def fetch(symbol, rng):
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
    p.add_argument("--range", default="5Y", help="History window: 1M/6M/1Y/3Y/5Y/10Y (default 5Y).")
    p.add_argument("--outdir", default="data", help="Directory to write <SYMBOL>.csv into.")
    args = p.parse_args()

    print(f"Fetching {len(args.symbols)} symbol(s), range={args.range}")
    failures = []
    for sym in args.symbols:
        try:
            rows = fetch(sym, args.range)
            out = os.path.join(args.outdir, f"{sym.upper()}.csv")
            write_csv(rows, out)
            print(f"  -> wrote {out}")
        except Exception as e:  # noqa: BLE001
            print(f"  !! {sym} failed: {e}", file=sys.stderr)
            failures.append(sym)
    if failures:
        sys.exit(f"Failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
