"""Index-only audit of complete predispatch archives, 2023-01..2025-11."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import unquote
from urllib.request import Request, urlopen

from solaralpha.historical_pipeline import ROOT, _write_json_atomic


OUT = ROOT / "data" / "derived" / "aemo_predisp_all_archive_index.json"
TABLES = ("PREDISPATCHPRICE", "PREDISPATCHREGIONSUM")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "a" and "href" in values:
            self.hrefs.append(values["href"])


def inspect_month(year, month, *, opener=urlopen):
    url = ("https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
           f"{year}/MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/"
           "PREDISP_ALL_DATA/")
    with opener(Request(url, headers={"User-Agent": "SolarAlpha archive audit/1.0"}),
                timeout=60) as response:
        content = response.read(500_000)
        if len(content) == 500_000:
            raise ValueError(f"Oversize index {url}")
    parser = Links()
    parser.feed(content.decode("utf-8", "replace"))
    names = [unquote(href.rsplit("/", 1)[-1]) for href in parser.hrefs]
    tables = {}
    for table in TABLES:
        tables[table] = [name for name in names if table in name
                         and "SENSITIVIT" not in name and name.endswith(".zip")]
    return {"month": f"{year}-{month:02d}", "index_url": url, "tables": tables}


def run():
    months = [(year, month) for year in (2023, 2024, 2025)
              for month in range(1, 13) if year != 2025 or month <= 11]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda item: inspect_month(*item), months))
    output = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "index-only, 2023-01..2025-11; no archived table payloads",
              "months": results}
    _write_json_atomic(OUT, output)
    return output


if __name__ == "__main__":
    result = run()
    for table in TABLES:
        print(table, sum(bool(month["tables"][table]) for month in result["months"]),
              "of", len(result["months"]), "months")
