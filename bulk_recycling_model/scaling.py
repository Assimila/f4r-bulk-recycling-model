import enum
import itertools

import networkx as nx
import numpy as np


class UnitSystem(enum.StrEnum):
    natural = "natural"  # natural units of the DAO dataset
    SI = "SI"
    scaled = "scaled"  # of order ~ 1, used by the model for numerical stability


class Scaling:
    def __init__(self, domain_length_scale: float = 22.25e5):
        """
        Args:
            domain_length_scale: Length scale of the domain in meters.
        """
        self.distance = DistanceScaling(domain_length_scale)
        self.evaporation = self.precipitation = EPScaling(domain_length_scale)
        self.water_vapor_flux = WaterVaporFluxScaling(domain_length_scale)


class VariableScaling:
    """
    Conversion of a single variable between systems of units.
    """

    def __init__(self, domain_length_scale: float):
        """
        Args:
            domain_length_scale: Length scale of the domain in meters.
        """
        self.conversion_factors: nx.DiGraph = nx.DiGraph()
        # add nodes for UnitSystem
        for unit in UnitSystem:
            self.conversion_factors.add_node(unit)

    def add_conversion(
        self,
        from_units: UnitSystem,
        to_units: UnitSystem,
        factor: float,
    ) -> None:
        """
        Add a conversion factor between two systems of units.

        The weight of the edge is the conversion factor such that
        value (to_units) = value (from_units) * factor.
        """
        if from_units == to_units:
            return
        self.conversion_factors.add_edge(from_units, to_units, factor=factor)
        self.conversion_factors.add_edge(to_units, from_units, factor=1.0 / factor)

    def get_factor(self, from_units: UnitSystem, to_units: UnitSystem) -> float:
        """
        Get the conversion factor between two systems of units.

        The weight of the edge is the conversion factor such that
        value (to_units) = value (from_units) * factor.
        """
        if from_units == to_units:
            return 1.0
        try:
            path = nx.shortest_path(self.conversion_factors, from_units, to_units)
        except nx.NetworkXNoPath:
            raise ValueError(f"No conversion from {from_units} to {to_units}")
        factor = 1.0
        for n1, n2 in itertools.pairwise(path):
            factor *= self.conversion_factors.edges[n1, n2]["factor"]
        return factor

    def convert[T: (float, np.ndarray)](self, value: T, from_units: UnitSystem, to_units: UnitSystem) -> T:
        """
        Convert a value from one system of units to another.
        """
        factor = self.get_factor(from_units, to_units)
        return value * factor


class EPScaling(VariableScaling):
    """
    Scaling for evaporation and precipitation.

    - natural units: mm/day
    - SI units: m/s
    - scaled units: same as natural units
    """

    def __init__(self, domain_length_scale: float):
        super().__init__(domain_length_scale)
        self.add_conversion(UnitSystem.natural, UnitSystem.SI, 1e-3 / 24 / 60 / 60)  # mm/day to m/s
        self.add_conversion(UnitSystem.natural, UnitSystem.scaled, 1.0)


class DistanceScaling(VariableScaling):
    """
    Scaling for distance.
    
    - natural units: undefined
    - SI units: m
    - scaled units: scaled to domain length scale
    """

    def __init__(self, domain_length_scale: float):
        super().__init__(domain_length_scale)
        self.add_conversion(UnitSystem.SI, UnitSystem.scaled, 1.0 / domain_length_scale)


class WaterVaporFluxScaling(VariableScaling):
    """
    Scaling for water vapor flux.

    - natural units: mb x m/s
    - SI units: m^2/s
    - scaled units: absorbs most of the scaling from other variables
    """

    def __init__(self, domain_length_scale: float):
        super().__init__(domain_length_scale)
        self.add_conversion(UnitSystem.natural, UnitSystem.SI, 1.02e-2)  # mb x m/s to m^2/s
        self.add_conversion(UnitSystem.SI, UnitSystem.scaled, 1e3 * 24 * 60 * 60 / domain_length_scale)
