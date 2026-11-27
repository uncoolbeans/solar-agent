"""Decision-time invariants for offline AEMO forecast-vintage research."""

import io
import unittest
from datetime import datetime, timedelta, timezone
from zipfile import ZipFile

from solaralpha.aemo_forecast_archive import (Forecast, aest_to_utc,
                                              interval_start_utc, latest_available,
                                              read_forecasts)


UTC = timezone.utc


def forecast(run_minute, target_minute, price=0.1, intervention=0):
    base = datetime(2025, 1, 1, tzinfo=UTC)
    return Forecast("P5MIN_REGIONSOLUTION", base + timedelta(minutes=run_minute),
                    base + timedelta(minutes=target_minute), price, intervention, {})


class AemoForecastArchiveTests(unittest.TestCase):
    def test_fixed_aest_and_interval_end_conversion(self):
        self.assertEqual(aest_to_utc("2025/01/20 12:05:00"),
                         datetime(2025, 1, 20, 2, 5, tzinfo=UTC))
        self.assertEqual(interval_start_utc("2025/01/20 12:05:00", 5),
                         datetime(2025, 1, 20, 2, 0, tzinfo=UTC))
        self.assertEqual(interval_start_utc("2025/06/20 12:30:00", 30),
                         datetime(2025, 6, 20, 2, 0, tzinfo=UTC))

    def test_parser_preserves_runs_and_excludes_other_regions(self):
        content = ("C,source\n"
                   "I,P5MIN,REGIONSOLUTION,9,RUN_DATETIME,INTERVAL_DATETIME,REGIONID,RRP,INTERVENTION\n"
                   "D,P5MIN,REGIONSOLUTION,9,2025/01/20 12:05:00,2025/01/20 12:10:00,NSW1,-100,0\n"
                   "D,P5MIN,REGIONSOLUTION,9,2025/01/20 12:10:00,2025/01/20 12:10:00,NSW1,-80,0\n"
                   "D,P5MIN,REGIONSOLUTION,9,2025/01/20 12:10:00,2025/01/20 12:10:00,VIC1,200,0\n")
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("sample.csv", content)
        buffer.seek(0)
        rows = list(read_forecasts("P5MIN_REGIONSOLUTION", buffer))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].target_start_utc, rows[1].target_start_utc)
        self.assertNotEqual(rows[0].run_utc, rows[1].run_utc)
        self.assertEqual(rows[0].price_aud_per_kwh, -0.1)

    def test_latest_vintage_and_publication_delay(self):
        rows = [forecast(-5, 60, 0.1), forecast(0, 60, 0.2),
                forecast(5, 60, 9.9)]
        decision = datetime(2025, 1, 1, tzinfo=UTC)
        target = decision + timedelta(hours=1)
        self.assertEqual(latest_available(rows, decision, target).price_aud_per_kwh, 0.2)
        self.assertEqual(latest_available(rows, decision, target,
                                          delay=timedelta(minutes=1)).price_aud_per_kwh, 0.1)
        # Changing a future vintage cannot change the earlier decision.
        rows[-1] = forecast(5, 60, -999)
        self.assertEqual(latest_available(rows, decision, target).price_aud_per_kwh, 0.2)

    def test_intervention_and_duplicate_handling(self):
        decision = datetime(2025, 1, 1, tzinfo=UTC)
        target = decision + timedelta(hours=1)
        ordinary, intervention = forecast(0, 60, 0.1), forecast(0, 60, 8, 1)
        self.assertEqual(latest_available([ordinary, intervention], decision, target), ordinary)
        with self.assertRaises(ValueError):
            latest_available([ordinary, ordinary], decision, target)

    def test_twelve_periods_do_not_cover_exact_one_hour_target(self):
        decision = datetime(2025, 1, 1, tzinfo=UTC)
        target = decision + timedelta(hours=1)
        # Latest conservatively available run at the decision ends its first interval at t.
        rows = [forecast(0, -5 + 5 * period) for period in range(12)]
        self.assertIsNone(latest_available(rows, decision, target))
        self.assertEqual(rows[-1].target_start_utc, decision + timedelta(minutes=50))

    def test_monthly_predispatch_snapshot_rejected_as_vintage(self):
        with self.assertRaisesRegex(ValueError, "do not preserve run vintages"):
            list(read_forecasts("PREDISPATCHPRICE"))


if __name__ == "__main__":
    unittest.main()
