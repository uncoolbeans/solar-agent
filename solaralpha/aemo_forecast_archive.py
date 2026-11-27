"""Offline AEMO MMSDM forecast-vintage smoke test; never imported by agents.

The monthly SQLLoader archives preserve forecast runs, but do not establish
when a particular run became publicly available. All joins therefore require
an explicit, assumed publication delay and report that assumption.
"""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from solaralpha.historical_pipeline import AEST, ROOT, _write_json_atomic


RAW = ROOT / "data" / "raw" / "aemo_forecast_2025_01"
MANIFEST = RAW / "manifest.json"
BASE = ("https://www.nemweb.com.au/Data_Archive/Wholesale_Electricity/"
        "MMSDM/2025/MMSDM_2025_01/MMSDM_Historical_Data_SQLLoader/DATA/")
TABLES = ("P5MIN_REGIONSOLUTION", "PREDISPATCHPRICE", "PREDISPATCHREGIONSUM")
MAX_BYTES = 60_000_000
UTC = timezone.utc


def source_url(table):
    if table not in TABLES:
        raise ValueError(table)
    return BASE + f"PUBLIC_ARCHIVE%23{table}%23FILE01%23202501010000.zip"


def source_path(table):
    return RAW / f"{table}.zip"


def fetch_tables(tables=TABLES, *, opener=urlopen):
    """Stream and hash bounded archives; never overwrite an existing snapshot."""
    manifest = (json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists()
                else {"schema_version": 1, "files": {}})
    for table in tables:
        url, path = source_url(table), source_path(table)
        entry = manifest["files"].get(table)
        if entry and entry["url"] != url:
            raise ValueError(f"URL changed for {table}")
        if path.exists():
            digest = sha256()
            size = 0
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    size += len(chunk)
                    digest.update(chunk)
            if not entry or digest.hexdigest() != entry["sha256"] or size != entry["bytes"]:
                raise ValueError(f"Untracked or changed raw archive: {path}")
            continue
        request = Request(url, headers={"User-Agent": "SolarAlpha forecast research/1.0"})
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".zip.tmp")
        digest = sha256()
        size = 0
        try:
            with opener(request, timeout=120) as response, temp.open("wb") as output:
                headers = {key: response.headers.get(key) for key in
                           ("Content-Length", "Last-Modified", "ETag", "Content-Type")}
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError(f"Archive exceeds {MAX_BYTES} bytes: {table}")
                    output.write(chunk)
                    digest.update(chunk)
            with ZipFile(temp) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Invalid ZIP member: {table}")
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        manifest["files"][table] = {
            "url": url, "sha256": digest.hexdigest(), "bytes": size,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "response_headers": headers, "path": path.relative_to(ROOT).as_posix(),
        }
        _write_json_atomic(MANIFEST, manifest)
    return manifest


def aest_to_utc(text):
    """AEMO market timestamps are fixed AEST interval ends, not Sydney DST time."""
    return datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S").replace(
        tzinfo=AEST).astimezone(UTC)


def interval_start_utc(end_text, minutes):
    return aest_to_utc(end_text) - timedelta(minutes=minutes)


@dataclass(frozen=True)
class Forecast:
    table: str
    run_utc: datetime
    target_start_utc: datetime
    price_aud_per_kwh: float | None
    intervention: int | None
    fields: dict


def read_forecasts(table, path=None, *, region="NSW1", start=None, end=None):
    """Read NSW forecast rows while retaining every run vintage and other fields.

    `start`/`end` filter target interval starts in UTC, not run timestamps.
    CSV I records define the schema; D records carry values. Each archive may
    contain several CSV members. A malformed or changed schema fails closed.
    """
    if table != "P5MIN_REGIONSOLUTION":
        raise ValueError("Monthly predispatch snapshots do not preserve run vintages")
    path = source_path(table) if path is None else path
    period_minutes = 5
    with ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            with archive.open(member) as file:
                import io
                rows = csv.reader(io.TextIOWrapper(file, encoding="utf-8-sig", newline=""))
                columns = None
                for row in rows:
                    if len(row) < 4 or (row[1], row[2]) != ("P5MIN", "REGIONSOLUTION"):
                        continue
                    if row[0] == "I":
                        columns = row[4:]
                        required = {"RUN_DATETIME", "INTERVAL_DATETIME", "REGIONID",
                                    "RRP", "INTERVENTION"}
                        if not required.issubset(columns):
                            raise ValueError(f"Missing required columns in {table}: {required-set(columns)}")
                    elif row[0] == "D":
                        if columns is None:
                            raise ValueError(f"Data before header in {member}")
                        if len(row[4:]) != len(columns):
                            raise ValueError(f"Column count changed in {member}")
                        values = dict(zip(columns, row[4:]))
                        if values["REGIONID"] != region:
                            continue
                        target = interval_start_utc(values["INTERVAL_DATETIME"], period_minutes)
                        if (start is not None and target < start) or (end is not None and target >= end):
                            continue
                        price = values.get("RRP")
                        intervention = values.get("INTERVENTION")
                        yield Forecast(table, aest_to_utc(values["RUN_DATETIME"]), target,
                                       float(price) / 1000 if price not in (None, "") else None,
                                       int(intervention) if intervention not in (None, "") else None,
                                       values)


def latest_available(forecasts, decision_utc, target_start_utc, *, delay=timedelta(),
                     intervention=0):
    """Select latest eligible vintage; later runs can never backfill earlier decisions.

    `run_utc + delay` is only a sensitivity proxy for actual publication time.
    A production join requires measured publication/receipt timestamps.
    """
    if delay < timedelta():
        raise ValueError("Publication delay must be nonnegative")
    eligible = [f for f in forecasts if f.target_start_utc == target_start_utc
                and f.run_utc + delay <= decision_utc
                and f.intervention == intervention]
    if not eligible:
        return None
    newest_run = max(f.run_utc for f in eligible)
    newest = [f for f in eligible if f.run_utc == newest_run]
    if len(newest) != 1:
        raise ValueError("Duplicate forecast vintage for target and intervention")
    return newest[0]
