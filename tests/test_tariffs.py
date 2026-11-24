"""Unit checks for the two illustrative per-kWh tariff calculations."""

import unittest

from solaralpha.tariffs import ILLUSTRATIVE_FIXED, ILLUSTRATIVE_SPOT_LINKED


class TariffTests(unittest.TestCase):
    def test_fixed_rates_do_not_change_with_wholesale_price(self):
        for spot in (-0.04829, 0.14488, 17.50):
            with self.subTest(spot=spot):
                self.assertEqual(ILLUSTRATIVE_FIXED.prices(spot), (0.30, 0.05))

    def test_spot_linked_rates_include_negative_prices_and_spikes(self):
        buy, sell = ILLUSTRATIVE_SPOT_LINKED.prices(-0.04829)
        self.assertAlmostEqual(buy, 0.12171)
        self.assertAlmostEqual(sell, -0.05829)

        buy, sell = ILLUSTRATIVE_SPOT_LINKED.prices(17.50)
        self.assertAlmostEqual(buy, 17.67)
        self.assertAlmostEqual(sell, 17.49)


if __name__ == "__main__":
    unittest.main()
