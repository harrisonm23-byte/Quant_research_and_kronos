"""Replication of harrisonm23-byte/Crypto_Data_Project on Alpaca crypto data (2021+).
Rules unchanged from their README: vol >= 2x trailing 30d mean = spike day;
red spike -> fwd close-to-close H=2; green spike -> H=10; excess vs unconditional.
De-overlapped per-coin trade backtest net 20bps. Core = their coins on Alpaca
(BTC/ETH/DOGE/LINK); extension = 6 Alpaca coins outside their universe.
Result 2026-07-11: core down t=3.05 / up t=2.25 (replicates); extension down t=1.39,
up DEAD -> deploy down-spike H2 only. Env: ALPACA_API_KEY/SECRET. See CODEX.md IV.
"""

import json, os, urllib.parse, urllib.request
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
KEY=os.environ["ALPACA_API_KEY"]; SEC=os.environ["ALPACA_API_SECRET"]
end=(datetime.now(timezone.utc)-timedelta(minutes=16)).strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch(sym):
    out=[]; page=None
    while True:
        p={"symbols":sym,"timeframe":"1Day","start":"2021-01-01","end":end,"limit":"1000"}
        if page: p["page_token"]=page
        url="https://data.alpaca.markets/v1beta3/crypto/us/bars?"+urllib.parse.urlencode(p)
        r=urllib.request.Request(url,headers={"APCA-API-KEY-ID":KEY,"APCA-API-SECRET-KEY":SEC})
        with urllib.request.urlopen(r,timeout=30) as resp: d=json.loads(resp.read())
        out += d.get("bars",{}).get(sym,[])
        page = d.get("next_page_token")
        if not page: break
    df=pd.DataFrame(out)
    df["t"]=pd.to_datetime(df["t"]); df=df.sort_values("t").reset_index(drop=True)
    return df

coins=["BTC/USD","ETH/USD","DOGE/USD","LINK/USD","AVAX/USD","SHIB/USD","LTC/USD","BCH/USD","UNI/USD","AAVE/USD"]
frames={}
for s in coins:
    try:
        df=fetch(s)
        if len(df)>400: frames[s]=df
    except Exception: pass
print("universe:", {k:len(v) for k,v in frames.items()})

def study(universe, label):
    rows=[]; ev_down=[]; ev_up=[]; base2=[]; base10=[]
    for s in universe:
        df=frames[s].copy()
        c=df["c"].values; o=df["o"].values; v=df["v"].values
        vm=pd.Series(v).rolling(30).mean().values
        vr=v/vm
        f2=np.full(len(df),np.nan); f10=np.full(len(df),np.nan)
        f2[:-2]=c[2:]/c[:-2]-1
        f10[:-10]=c[10:]/c[:-10]-1
        for i in range(30,len(df)):
            if not np.isnan(f2[i]): base2.append(f2[i])
            if not np.isnan(f10[i]): base10.append(f10[i])
            if np.isnan(vr[i]) or vr[i]<2.0: continue
            if c[i]<o[i] and not np.isnan(f2[i]): ev_down.append(f2[i])
            if c[i]>=o[i] and not np.isnan(f10[i]): ev_up.append(f10[i])
        for name,H,cond in [("down",2,lambda i: c[i]<o[i]),("up",10,lambda i: c[i]>=o[i])]:
            last=-999
            for i in range(30,len(df)-H):
                if np.isnan(vr[i]) or vr[i]<2.0 or not cond(i): continue
                if i-last<H: continue
                last=i
                rows.append(dict(coin=s,sig=name,ret=c[i+H]/c[i]-1-0.002))
    T=pd.DataFrame(rows)
    d=np.array(ev_down); u=np.array(ev_up); b2=np.array(base2); b10=np.array(base10)
    def welch(a,b):
        return (a.mean()-b.mean())/np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
    print(f"\n===== {label} =====")
    print(f"DOWN-spike H2: n={len(d)} excess {(d.mean()-b2.mean())*100:+.2f}% t={welch(d,b2):.2f}")
    print(f"UP-spike H10:  n={len(u)} excess {(u.mean()-b10.mean())*100:+.2f}% t={welch(u,b10):.2f}")
    for sig in ["down","up"]:
        S=T[T.sig==sig]
        if len(S):
            a=S["ret"].values
            print(f"  trades {sig}: n={len(a)} mean net {a.mean()*100:+.2f}% median {np.median(a)*100:+.2f}% WR {(a>0).mean():.0%}")

core=[s for s in ["BTC/USD","ETH/USD","DOGE/USD","LINK/USD"] if s in frames]
study(core, "REPLICATION CORE (their coins on Alpaca, 2021+)")
ext=[s for s in frames if s not in core]
if ext: study(ext, "EXTENSION (out-of-universe coins)")
