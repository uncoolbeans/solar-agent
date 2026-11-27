"""Check official MMSDM monthly index availability; download no archive payloads."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import unquote
from urllib.request import Request, urlopen

from solaralpha.aemo_forecast_archive import ROOT, TABLES
from solaralpha.historical_pipeline import _write_json_atomic


OUT = ROOT / "data" / "derived" / "aemo_forecast_archive_index.json"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "a" and "href" in attributes:
            self.hrefs.append(attributes["href"])


def inspect_month(year, month, *, opener=urlopen):
    url = ("https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
           f"{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/DATA/")
    with opener(Request(url, headers={"User-Agent": "SolarAlpha forecast research/1.0"}),
                timeout=60) as response:
        page = response.read(2_000_000)
        if len(page) == 2_000_000:
            raise ValueError(f"Index too large: {url}")
    parser = Links()
    parser.feed(page.decode("utf-8", errors="replace"))
    filenames = [PurePosixPath(unquote(href)).name for href in parser.hrefs]
    found = {}
    for table in TABLES:
        candidates = [name for name in filenames if name.lower().endswith(".zip")
                      and (f"#{table}#" in name or f"_{table}_" in name)
                      and "SENSITIVIT" not in name]
        found[table] = candidates
    return {"month": f"{year}-{month:02d}", "index_url": url,
            "tables": found}


def run():
    months = [(year, month) for year in (2023, 2024, 2025)
              for month in range(1, 13) if year != 2025 or month <= 11]
    with ThreadPoolExecutor(max_workers=5) as pool:
        entries = list(pool.map(lambda ym: inspect_month(*ym), months))
    payload = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "2023-01 through 2025-11; index listing only",
               "months": entries}
    _write_json_atomic(OUT, payload)
    return payload


if __name__ == "__main__":
    result = run()
    for table in TABLES:
        print(table, sum(bool(row["tables"][table]) for row in result["months"]),
              "of", len(result["months"]), "monthly indexes")
