"""Stage 2: fetch 1m bars (04:00-20:00 ET, SIP) for each gapper event day.
Appends to event_bars.csv with checkpointing (skips already-fetched events).
"""
import json, os, time, urllib.parse, urllib.request
import pandas as pd

KEY = os.environ["ALPACA_API_KEY"]; SEC = os.environ["ALPACA_API_SECRET"]
HDR = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SEC}
OUT = os.path.dirname(os.path.abspath(__file__))
EV = pd.read_csv(os.path.join(OUT, "gapper_events.csv"))
DEST = os.path.join(OUT, "event_bars.csv")

done = set()
if os.path.exists(DEST):
    d0 = pd.read_csv(DEST, usecols=["symbol", "date"]).drop_duplicates()
    done = set(zip(d0["symbol"], d0["date"]))
print(f"events {len(EV)}, already fetched {len(done)}", flush=True)

def get(url):
    for a in range(5):
        try:
            r = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(r, timeout=60) as resp:
                return json.loads(resp.read())
        except Exception:
            if a == 4:
                return None
            time.sleep(1.5 ** a)

first = not os.path.exists(DEST)
buf = []
cnt = 0
for _, e in EV.iterrows():
    key = (e["symbol"], e["date"])
    if key in done:
        continue
    p = {"timeframe": "1Min", "start": f"{e['date']}T08:00:00Z", "end": f"{e['date']}T23:59:00Z",
         "limit": "10000", "feed": "sip", "adjustment": "all"}
    d = get(f"https://data.alpaca.markets/v2/stocks/{e['symbol']}/bars?" + urllib.parse.urlencode(p))
    cnt += 1
    if d and d.get("bars"):
        df = pd.DataFrame(d["bars"])
        df["symbol"] = e["symbol"]; df["date"] = e["date"]
        buf.append(df)
    if len(buf) >= 40:
        pd.concat(buf).to_csv(DEST, mode="a", header=first, index=False)
        first = False; buf = []
        print(f"  fetched {cnt} events...", flush=True)
if buf:
    pd.concat(buf).to_csv(DEST, mode="a", header=first, index=False)
print(f"DONE stage 2: {cnt} events fetched", flush=True)
