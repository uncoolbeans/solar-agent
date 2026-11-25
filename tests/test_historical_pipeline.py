"""Time-zone, source-integrity, unit and partition checks for data ingestion."""

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import unittest

from solaralpha.historical_pipeline import (
    CSV_FIELDS, ROOT, Source, _aemo_rows, _weather_rows, build_month, ensure_sources,
    local_month_bounds, month_date, months_between, source_plan,
    split_for_local_date,
)
from solaralpha.pipeline_diagnostics import compare_development_week, read_month


class HistoricalPipelineTests(unittest.TestCase):
    def fixture(self, name):
        path = Path(__file__).with_name(f"_historical_pipeline_test_{name}")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.addCleanup(lambda: path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True))
        return path

    def test_january_plan_uses_previous_aest_month_and_pins_weather(self):
        sources = source_plan(months_between(month_date("2025-01"),
                                             month_date("2025-01")))
        names = {source.path.name for source in sources}
        self.assertEqual(names, {
            "PRICE_AND_DEMAND_202412_NSW1.CSV",
            "PRICE_AND_DEMAND_202501_NSW1.CSV",
            "open_meteo_era5_sydney_local_202501.json",
        })
        weather = next(source for source in sources if source.kind == "open_meteo_reanalysis")
        self.assertIn("models=era5", weather.url)
        self.assertIn("start_date=2024-12-31", weather.url)
        self.assertIn("timezone=GMT", weather.url)

    def test_daylight_saving_month_lengths_and_repeated_clock_hour(self):
        april_start, april_end = local_month_bounds(date(2025, 4, 1))
        october_start, october_end = local_month_bounds(date(2025, 10, 1))
        self.assertEqual((april_end-april_start)/timedelta(minutes=5), 8652)
        self.assertEqual((october_end-october_start)/timedelta(minutes=5), 8916)
        from solaralpha.historical_pipeline import sydney_zone
        zone = sydney_zone()
        first = datetime(2025, 4, 5, 15, 30, tzinfo=timezone.utc).astimezone(zone)
        second = datetime(2025, 4, 5, 16, 30, tzinfo=timezone.utc).astimezone(zone)
        self.assertEqual((first.hour, second.hour), (2, 2))
        self.assertNotEqual(first.utcoffset(), second.utcoffset())

    def test_split_boundaries_keep_development_week_out_of_scores(self):
        self.assertEqual(split_for_local_date(date(2024, 12, 31)), "train")
        self.assertEqual(split_for_local_date(date(2025, 1, 1)), "validation")
        self.assertEqual(split_for_local_date(date(2025, 11, 30)), "validation")
        self.assertEqual(split_for_local_date(date(2025, 12, 1)), "held_out_test")
        self.assertEqual(split_for_local_date(date(2026, 2, 28)), "held_out_test")
        self.assertTrue(all(name.startswith("realised_")
                            for name in CSV_FIELDS[13:]))

    def test_duplicate_aemo_rows_and_invalid_demand_are_detected(self):
        path = self.fixture("sample.csv")
        path.write_text(
            "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n"
            "NSW1,2025/01/01 00:05:00,7000,-1000,TRADE\n"
            "NSW1,2025/01/01 00:05:00,7000,17500,TRADE\n"
            "NSW1,2025/01/01 00:10:00,7,50,TRADE\n",
            encoding="utf-8")
        start = datetime(2024, 12, 31, 14, tzinfo=timezone.utc)
        values, duplicates, invalid = _aemo_rows([path], start, start+timedelta(minutes=10))
        self.assertEqual(len(values), 1)
        self.assertEqual(values[start], (-1000, 7000))
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(invalid), 1)

    def test_weather_units_and_duplicate_hours_are_checked(self):
        payload = {"timezone": "GMT", "utc_offset_seconds": 0,
                   "hourly_units": {"temperature_2m": "°C",
                                    "shortwave_radiation": "W/m²",
                                    "cloud_cover": "%"},
                   "hourly": {"time": ["2025-01-01T00:00", "2025-01-01T00:00"],
                              "temperature_2m": [25, 26],
                              "shortwave_radiation": [100, 200],
                              "cloud_cover": [30, 40]}}
        path = self.fixture("weather.json")
        path.write_text(json.dumps(payload), encoding="utf-8")
        values, duplicates, invalid = _weather_rows(path)
        self.assertEqual((len(values), len(duplicates), len(invalid)), (1, 1, 0))
        payload["hourly_units"]["shortwave_radiation"] = "kW/m²"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "units"):
            _weather_rows(path)

    def test_cached_raw_file_is_never_overwritten_and_hash_checked(self):
        content = (b"REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n"
                   b"NSW1,2025/01/01 00:05:00,7000,100,TRADE\n")
        path = self.fixture("download.csv")
        manifest_path = self.fixture("manifest.json")
        source = Source(path, "https://example.invalid/sample.csv",
                        "aemo_price_and_demand")
        class Response:
            headers = {"Content-Type": "text/csv"}
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, _): return content
        calls = []
        def opener(*_args, **_kwargs):
            calls.append(1)
            return Response()
        ensure_sources([source], manifest_path, opener)
        ensure_sources([source], manifest_path, opener)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sha256(path.read_bytes()).hexdigest(),
                         json.loads(manifest_path.read_text())["files"]
                         [path.relative_to(ROOT).as_posix()]["sha256"])
        path.write_bytes(content+b"tampered")
        with self.assertRaisesRegex(ValueError, "manifest"):
            ensure_sources([source], manifest_path, opener)

    def test_missing_intervals_write_report_and_block_processed_csv(self):
        fixture_dir = Path(__file__).parent
        for name in ("PRICE_AND_DEMAND_202412_NSW1.CSV",
                     "PRICE_AND_DEMAND_202501_NSW1.CSV"):
            path = fixture_dir / name
            self.addCleanup(lambda path=path: path.unlink(missing_ok=True))
            path.write_text("REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE\n",
                            encoding="utf-8")
        weather_path = fixture_dir / "open_meteo_era5_sydney_local_202501.json"
        self.addCleanup(lambda: weather_path.unlink(missing_ok=True))
        weather_path.write_text(json.dumps({
            "timezone": "GMT", "utc_offset_seconds": 0,
            "hourly_units": {"temperature_2m": "°C",
                             "shortwave_radiation": "W/m²", "cloud_cover": "%"},
            "hourly": {"time": [], "temperature_2m": [],
                       "shortwave_radiation": [], "cloud_cover": []},
        }), encoding="utf-8")
        quality_path = fixture_dir / "2025-01_quality.json"
        self.addCleanup(lambda: quality_path.unlink(missing_ok=True))
        with self.assertRaisesRegex(ValueError, "Incomplete/invalid"):
            build_month(date(2025, 1, 1), raw_dir=fixture_dir,
                        processed_dir=fixture_dir)
        report = json.loads(quality_path.read_text(encoding="utf-8"))
        self.assertEqual(report["missing_aemo_intervals"], 8928)
        self.assertEqual(report["missing_weather_intervals"], 8928)
        self.assertFalse((fixture_dir / "2025-01_realised.csv").exists())

    @unittest.skipUnless((ROOT / "data/processed/2025-01_realised.csv").exists(),
                         "January smoke-test output not present")
    def test_smoke_test_reuses_exact_price_and_tariff_week(self):
        rows = read_month()
        self.assertEqual(len(rows), 8928)
        comparison = compare_development_week(rows)
        self.assertTrue(comparison["timestamps_match"])
        for key in ("spot_aud_per_kwh", "illustrative_buy_aud_per_kwh",
                    "illustrative_sell_aud_per_kwh"):
            self.assertEqual(comparison["comparison"][key]["exact_matches"], 2016)
        self.assertEqual(sum(row["include_in_validation_score"] == "1" for row in rows),
                         6912)


if __name__ == "__main__":
    unittest.main()
