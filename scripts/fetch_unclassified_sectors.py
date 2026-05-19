import ssl
import urllib.request
ssl._create_default_https_context = ssl._create_unverified_context

import requests
requests.packages.urllib3.disable_warnings()
_orig_request = requests.Session.request
def _no_verify(self, method, url, **kwargs):
    kwargs.setdefault("verify", False)
    return _orig_request(self, method, url, **kwargs)
requests.Session.request = _no_verify

import yfinance as yf
import json

SYMBOLS = [
    "133A.T","140A.T","159A.T","162A.T","163A.T","170A.T","178A.T","179A.T",
    "180A.T","181A.T","182A.T","183A.T","188A.T","200A.T","201A.T","210A.T",
    "213A.T","221A.T","223A.T","224A.T","233A.T","234A.T","235A.T","236A.T",
    "237A.T","238A.T","257A.T","258A.T","273A.T","282A.T","283A.T","294A.T",
    "295A.T","313A.T","314A.T","315A.T","316A.T","318A.T","328A.T","345A.T",
    "346A.T","348A.T","349A.T","354A.T","356A.T","360A.T","363A.T","364A.T",
    "376A.T","379A.T","380A.T","381A.T","382A.T","383A.T","392A.T","394A.T",
    "395A.T","396A.T","399A.T","401A.T","404A.T","408A.T","412A.T","413A.T",
    "424A.T","425A.T","426A.T","435A.T","443A.T","447A.T","448A.T","449A.T",
    "450A.T","451A.T","452A.T","453A.T","459A.T","461A.T","465A.T","466A.T",
    "467A.T","468A.T","8301.T","8421.T",
]

results = []
for sym in SYMBOLS:
    try:
        info = yf.Ticker(sym).info
        sector = info.get("sector") or ""
        industry = info.get("industry") or ""
        name = info.get("longName") or info.get("shortName") or ""
        results.append({"symbol": sym, "name": name, "sector": sector, "industry": industry})
        print(f"{sym}: {name} | {sector}")
    except Exception as e:
        results.append({"symbol": sym, "name": "", "sector": "", "industry": str(e)})
        print(f"{sym}: ERROR {e}")

out = "unclassified_sectors.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Done -> {out}")
