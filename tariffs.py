"""Illustrative per-kWh household tariffs for offline comparisons.

Spot input and returned buy/sell prices are all in AUD/kWh. These examples
are not a quote or a model of any particular retailer's contract.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyTariff:
    name: str
    import_base: float
    export_base: float
    import_spot_multiplier: float = 0.0
    export_spot_multiplier: float = 0.0

    def prices(self, spot_aud_kwh):
        """Return (household buy price, sell price) in AUD/kWh."""
        return (self.import_base + self.import_spot_multiplier * spot_aud_kwh,
                self.export_base + self.export_spot_multiplier * spot_aud_kwh)


ILLUSTRATIVE_FIXED = EnergyTariff("fixed", import_base=0.30, export_base=0.05)
ILLUSTRATIVE_SPOT_LINKED = EnergyTariff(
    "spot_linked", import_base=0.17, export_base=-0.01,
    import_spot_multiplier=1.0, export_spot_multiplier=1.0,
)
EXAMPLE_TARIFFS = (ILLUSTRATIVE_FIXED, ILLUSTRATIVE_SPOT_LINKED)
