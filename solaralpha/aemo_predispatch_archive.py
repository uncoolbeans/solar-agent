"""Offline reader for complete January 2025 AEMO predispatch vintages.

This archive is for diagnostics. Nominal run time and LASTCHANGED are not
verified public release or receipt timestamps.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from solaralpha.aemo_forecast_archive import UTC, aest_to_utc
from solaralpha.historical_pipeline import ROOT, _write_json_atomic


RAW = ROOT / "data" / "raw" / "aemo_predisp_all_2025_01"
MANIFEST = RAW / "manifest.json"
BASE = ("https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/"
        "2025/MMSDM_2025_01/MMSDM_Historical_Data_SQLLoader/PREDISP_ALL_DATA/")
TABLES = ("PREDISPATCHPRICE", "PREDISPATCHREGIONSUM")
MAX_BYTES = 60_000_000


def source_url(table):
    if table not in TABLES:
        raise ValueError(table)
    return BASE + f"PUBLIC_ARCHIVE%23{table}%23ALL%23FILE01%23202501010000.zip"


def source_path(table):
    return RAW / f"{table}_ALL.zip"


def fetch_tables(tables=TABLES, *, opener=urlopen):
    """Download bounded ZIPs once, then verify the immutable cached bytes."""
    manifest = (json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists()
                else {"schema_version": 1, "files": {}})
    for table in tables:
        url, path = source_url(table), source_path(table)
        old = manifest["files"].get(table)
        if old and old["url"] != url:
            raise ValueError(f"Source URL changed: {table}")
        digest = sha256()
        size = 0
        if path.exists():
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if not old or old["sha256"] != digest.hexdigest() or old["bytes"] != size:
                raise ValueError(f"Untracked or altered source: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".zip.tmp")
        try:
            request = Request(url, headers={"User-Agent": "SolarAlpha predispatch audit/1.0"})
            with opener(request, timeout=120) as response, temp.open("wb") as output:
                headers = {key: response.headers.get(key) for key in
                           ("Content-Length", "Last-Modified", "ETag", "Content-Type")}
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError(f"Source exceeds {MAX_BYTES} bytes: {table}")
                    digest.update(chunk)
                    output.write(chunk)
            with ZipFile(temp) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Corrupt ZIP member: {table}")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        manifest["files"][table] = {
            "url": url, "path": path.relative_to(ROOT).as_posix(),
            "bytes": size, "sha256": digest.hexdigest(),
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "response_headers": headers,
        }
        _write_json_atomic(MANIFEST, manifest)
    return manifest


def nominal_run_utc(sequence):
    """Decode YYYYMMDDSS with SS=01 at 04:30 fixed AEST, then 30-min steps."""
    if len(sequence) != 10 or not sequence.isdigit():
        raise ValueError(f"Unexpected PREDISPATCHSEQNO: {sequence}")
    number = int(sequence[-2:])
    if not 1 <= number <= 48:
        raise ValueError(f"Unexpected PREDISPATCHSEQNO slot: {sequence}")
    day = datetime.strptime(sequence[:8], "%Y%m%d")
    nominal = day + timedelta(hours=4, minutes=30 * number)
    return aest_to_utc(nominal.strftime("%Y/%m/%d %H:%M:%S"))


@dataclass(frozen=True)
class PredispatchForecast:
    table: str
    sequence: str
    nominal_run_utc: datetime
    lastchanged_utc: datetime | None
    target_start_utc: datetime
    price_aud_per_kwh: float | None
    intervention: int
    runno: int
    fields: dict


def read_forecasts(table, path=None, *, region="NSW1", target_start=None,
                   target_end=None):
    """Stream each forecast vintage; never collapse runs by target interval."""
    if table not in TABLES:
        raise ValueError(table)
    archive_path = source_path(table) if path is None else path
    expected_subtable = ("REGION_PRICES" if table == "PREDISPATCHPRICE"
                         else "REGION_SOLUTION")
    with ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            with archive.open(member) as file:
                reader = csv.reader(io.TextIOWrapper(file, encoding="utf-8-sig", newline=""))
                columns = None
                for row in reader:
                    if len(row) < 4 or (row[1], row[2]) != ("PREDISPATCH", expected_subtable):
                        continue
                    if row[0] == "I":
                        columns = row[4:]
                        required = {"PREDISPATCHSEQNO", "RUNNO", "REGIONID",
                                    "INTERVENTION", "DATETIME", "LASTCHANGED"}
                        if not required.issubset(columns):
                            raise ValueError(f"Changed schema: {table} {required - set(columns)}")
                    elif row[0] == "D":
                        if columns is None or len(row[4:]) != len(columns):
                            raise ValueError(f"Malformed row in {member}")
                        values = dict(zip(columns, row[4:]))
                        if values["REGIONID"] != region:
                            continue
                        target = aest_to_utc(values["DATETIME"]) - timedelta(minutes=30)
                        if ((target_start is not None and target < target_start)
                                or (target_end is not None and target >= target_end)):
                            continue
                        lastchanged = values["LASTCHANGED"]
                        price = values.get("RRP")
                        yield PredispatchForecast(
                            table, values["PREDISPATCHSEQNO"],
                            nominal_run_utc(values["PREDISPATCHSEQNO"]),
                            aest_to_utc(lastchanged) if lastchanged else None,
                            target, float(price) / 1000 if price not in (None, "") else None,
                            int(values["INTERVENTION"]), int(values["RUNNO"]), values)


def latest_available(rows, decision_utc, target_start_utc, *, delay=timedelta(),
                     intervention=0, require_nominal_run_elapsed=False):
    """Research-only as-of selector using LASTCHANGED + assumed delay.

    LASTCHANGED is not certified publication time. Selection must be reported
    as a sensitivity case, never an established operational availability fact.
    """
    if delay < timedelta():
        raise ValueError("Delay must be nonnegative")
    eligible = [row for row in rows if row.target_start_utc == target_start_utc
                and row.intervention == intervention and row.lastchanged_utc is not None
                and row.lastchanged_utc + delay <= decision_utc
                and (not require_nominal_run_elapsed
                     or row.nominal_run_utc <= decision_utc)]
    if not eligible:
        return None
    newest_key = max((row.nominal_run_utc, row.runno, row.lastchanged_utc)
                     for row in eligible)
    newest = [row for row in eligible
              if (row.nominal_run_utc, row.runno, row.lastchanged_utc) == newest_key]
    if len(newest) != 1:
        raise ValueError("Ambiguous lastchanged vintage; reject duplicate")
    return newest[0]
