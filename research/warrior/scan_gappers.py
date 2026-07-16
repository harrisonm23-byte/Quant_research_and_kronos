"""Stage 1: historical small-cap gapper scanner (Alpaca daily bars).

Criteria (Warrior-canon, from the spec's small-cap track):
  gap = today's open / yesterday's close - 1 >= 7%
  yesterday close in [$1.50, $20]
  today dollar volume >= $5M
  top 8 per day by gap size
Window: 2023-06-01 -> now. Output: gapper_events.csv (symbol, date, gap, prev_close,
o/h/l/c/v, dollar_vol, rel_vol vs 30d median).
CAVEAT: current-listings universe only -> survivorship bias (delisted gappers missing).
"""
import json, os, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

KEY = os.environ["ALPACA_API_KEY"]; SEC = os.environ["ALPACA_API_SECRET"]
HDR = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}
OUT = os.path.dirname(os.path.abspath(__file__))
END = (datetime.now(timezone.utc) - timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")
START = "2023-06-01"


def get(url):
    for attempt in range(5):
        try:
            r = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(r, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


# 1. asset list: active, tradable US equities on major exchanges
assets = get("https://paper-api.alpaca.markets/v2/assets?status=active&asset_class=us_equity")
syms = [a["symbol"] for a in assets
        if a.get("tradable") and a.get("exchange") in ("NASDAQ", "NYSE", "AMEX", "ARCA", "BATS")
        and "." not in a["symbol"] and "/" not in a["symbol"] and len(a["symbol"]) <= 5]
syms = sorted(set(syms))
print(f"universe: {len(syms)} symbols", flush=True)

# 2. batch daily bars, compute gaps
events = []
B = 200
for bi in range(0, len(syms), B):
    batch = syms[bi:bi + B]
    page = None
    bars = {}
    while True:
        p = {"symbols": ",".join(batch), "timeframe": "1Day", "start": START, "end": END,
             "limit": "10000", "feed": "sip", "adjustment": "all"}
        if page:
            p["page_token"] = page
        d = get("https://data.alpaca.markets/v2/stocks/bars?" + urllib.parse.urlencode(p))
        for s, bs in (d.get("bars") or {}).items():
            bars.setdefault(s, []).extend(bs)
        page = d.get("next_page_token")
        if not page:
            break
    for s, bs in bars.items():
        if len(bs) < 35:
            continue
        df = pd.DataFrame(bs)
        c = df["c"].values; o = df["o"].values; v = df["v"].values
        pc = np.r_[np.nan, c[:-1]]
        gap = o / pc - 1
        dv = c * v
        vmed = pd.Series(v).rolling(30).median().values
        for i in range(30, len(df)):
            if not (0.07 <= gap[i] < 5):     # sanity cap: >400% gap = data artifact
                continue
            if not (1.5 <= pc[i] <= 20):
                continue
            if dv[i] < 5e6:
                continue
            events.append(dict(symbol=s, date=df["t"].iloc[i][:10], gap=round(gap[i], 4),
                               prev_close=pc[i], o=o[i], h=df["h"].iloc[i], l=df["l"].iloc[i],
                               c=c[i], v=int(v[i]), dollar_vol=int(dv[i]),
                               rel_vol=round(v[i] / vmed[i], 1) if vmed[i] > 0 else np.nan))
    if (bi // B) % 10 == 0:
        print(f"  batch {bi//B+1}/{(len(syms)+B-1)//B}, events so far {len(events)}", flush=True)

E = pd.DataFrame(events)
# top 8 per day by gap
E = E.sort_values(["date", "gap"], ascending=[True, False]).groupby("date").head(8).reset_index(drop=True)
E.to_csv(os.path.join(OUT, "gapper_events.csv"), index=False)
print(f"\nDONE: {len(E)} gapper events across {E['date'].nunique()} days "
      f"({len(E)/max(E['date'].nunique(),1):.1f}/day)")
print("gap distribution:", E["gap"].describe()[["mean", "50%", "max"]].round(3).to_dict())
print("sample:", E.head(5).to_dict("records"))
