"""Short-horizon and complete-predispatch decision-time safeguards."""

import io
from dataclasses import replace
import unittest
from datetime import datetime, timedelta, timezone
from zipfile import ZipFile

from solaralpha.aemo_forecast_archive import Forecast, latest_available as latest_p5
from solaralpha.aemo_horizon_diagnostic import half_hour_average
from solaralpha.aemo_predispatch_archive import (PredispatchForecast, latest_available,
                                                 nominal_run_utc, read_forecasts)


UTC = timezone.utc
BASE = datetime(2025, 1, 20, 0, tzinfo=UTC)


def pd_row(sequence="2025012001", lastchanged=-5, target=60, price=0.1,
           intervention=0):
    return PredispatchForecast("PREDISPATCHPRICE", sequence,
        BASE + timedelta(minutes=30), BASE + timedelta(minutes=lastchanged),
        BASE + timedelta(minutes=target), price, intervention, 1, {})


class AemoHorizonTests(unittest.TestCase):
    def test_predispatch_sequence_is_nominal_aest_run(self):
        self.assertEqual(nominal_run_utc("2025010101"),
                         datetime(2024, 12, 31, 18, 30, tzinfo=UTC))
        self.assertEqual(nominal_run_utc("2025010148"),
                         datetime(2025, 1, 1, 18, 0, tzinfo=UTC))
        with self.assertRaises(ValueError):
            nominal_run_utc("2025010100")

    def test_complete_archive_preserves_two_price_vintages(self):
        content = ("C,source\n"
            "I,PREDISPATCH,REGION_PRICES,2,PREDISPATCHSEQNO,RUNNO,REGIONID,PERIODID,INTERVENTION,RRP,LASTCHANGED,DATETIME\n"
            "D,PREDISPATCH,REGION_PRICES,2,2025012001,1,NSW1,01,0,100,2025/01/20 04:02:00,2025/01/20 05:30:00\n"
            "D,PREDISPATCH,REGION_PRICES,2,2025012002,1,NSW1,01,0,-50,2025/01/20 04:32:00,2025/01/20 05:30:00\n"
            "D,PREDISPATCH,REGION_PRICES,2,2025012002,1,VIC1,01,0,300,2025/01/20 04:32:00,2025/01/20 05:30:00\n")
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("price.csv", content)
        buffer.seek(0)
        rows = list(read_forecasts("PREDISPATCHPRICE", buffer))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].target_start_utc, rows[1].target_start_utc)
        self.assertNotEqual(rows[0].sequence, rows[1].sequence)
        self.assertEqual(rows[1].price_aud_per_kwh, -0.05)
        self.assertEqual(rows[0].target_start_utc,
                         datetime(2025, 1, 19, 19, tzinfo=UTC))

    def test_publication_delay_and_future_vintage_cannot_leak(self):
        target = BASE + timedelta(hours=1)
        earlier = pd_row(lastchanged=-10, price=0.1)
        later = pd_row(sequence="2025012002", lastchanged=2, price=0.2)
        self.assertEqual(latest_available([earlier, later], BASE, target), earlier)
        self.assertIsNone(latest_available([earlier, later], BASE, target,
                                            delay=timedelta(minutes=15)))
        changed_future = pd_row(sequence="2025012002", lastchanged=2, price=-999)
        self.assertEqual(latest_available([earlier, changed_future], BASE, target), earlier)

    def test_nominal_elapsed_intervention_and_duplicate_rules(self):
        target = BASE + timedelta(hours=1)
        ordinary = pd_row()
        intervention = pd_row(intervention=1, price=9)
        self.assertEqual(latest_available([ordinary, intervention], BASE, target), ordinary)
        self.assertIsNone(latest_available([ordinary], BASE, target,
                                           require_nominal_run_elapsed=True))
        with self.assertRaises(ValueError):
            latest_available([ordinary, ordinary], BASE, target)

    def test_newer_nominal_run_wins_over_late_revision_of_older_run(self):
        target = BASE + timedelta(hours=1)
        older = replace(pd_row(lastchanged=-1, price=0.1),
                        nominal_run_utc=BASE - timedelta(minutes=30))
        newer = replace(pd_row(sequence="2025012002", lastchanged=-2, price=0.2),
                        nominal_run_utc=BASE)
        self.assertEqual(latest_available([older, newer], BASE, target), newer)

    def test_half_hour_target_requires_six_exact_five_minute_values(self):
        prices = {BASE + timedelta(minutes=5*i): float(i) for i in range(6)}
        self.assertEqual(half_hour_average(prices, BASE), 2.5)
        del prices[BASE + timedelta(minutes=20)]
        self.assertIsNone(half_hour_average(prices, BASE))

    def test_p5_target_start_alignment_and_missing_coverage(self):
        decision = BASE
        near = Forecast("P5MIN_REGIONSOLUTION", BASE,
                        BASE + timedelta(minutes=45), 0.1, 0, {})
        future = Forecast("P5MIN_REGIONSOLUTION", BASE + timedelta(minutes=5),
                          BASE + timedelta(minutes=60), 0.2, 0, {})
        self.assertEqual(latest_p5([near], decision,
                                   BASE + timedelta(minutes=45)), near)
        self.assertIsNone(latest_p5([future], decision,
                                    BASE + timedelta(minutes=60)))


if __name__ == "__main__":
    unittest.main()
