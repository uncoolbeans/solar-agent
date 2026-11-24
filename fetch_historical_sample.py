"""Download a fixed public price and weather snapshot for an offline replay.

Run once: python fetch_historical_sample.py
The replay itself reads local files and makes no network requests.
"""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"
PRICE_FILE = RAW_DIR / "PRICE_AND_DEMAND_202501_NSW1.CSV"
WEATHER_FILE = RAW_DIR / "sydney_weather_2025-01-12_20.json"
PRICE_URL = ("https://www.nemweb.com.au/Reports/CURRENT/WebData/NEM/PRICEANDDEMAND/"
             "PRICE_AND_DEMAND_202501_NSW1.CSV")
WEATHER_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude=-33.87&longitude=151.21"
               "&start_date=2025-01-12&end_date=2025-01-20"
               "&hourly=shortwave_radiation,cloud_cover,temperature_2m&timezone=GMT")


def download(url, path):
    request = Request(url, headers={"User-Agent": "SolarAlpha historical replay/1.0"})
    with urlopen(request, timeout=30) as response:
        content = response.read(2_000_001)
    if not content or len(content) > 2_000_000:
        raise ValueError(f"Unexpected download size for {url}")
    if path == PRICE_FILE and not content.startswith(b"REGION,SETTLEMENTDATE,"):
        raise ValueError("AEMO response does not look like the expected CSV")
    if path == WEATHER_FILE and "hourly" not in json.loads(content):
        raise ValueError("Weather response has no hourly data")
    path.write_bytes(content)
    print(f"Saved {path} ({len(content)} bytes, SHA-256 {hashlib.sha256(content).hexdigest()})")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url, path in ((PRICE_URL, PRICE_FILE), (WEATHER_URL, WEATHER_FILE)):
        if path.exists():
            print(f"Already present: {path}")
        else:
            download(url, path)


if __name__ == "__main__":
    main()
