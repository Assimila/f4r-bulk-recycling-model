import unittest

from bulk_recycling_model import scaling


class TestVariableScaling(unittest.TestCase):

    def test_no_edges(self):
        obj = scaling.VariableScaling(domain_length_scale=1.0)
        # test that the 3 nodes are present
        assert {"natural", "SI", "scaled"} == set(obj.conversion_factors.nodes)
        # test that there are no edges
        assert len(obj.conversion_factors.edges) == 0

        with self.assertRaises(ValueError):
            obj.convert(1, "natural", "SI")

    def test_with_conversions(self):
        obj = scaling.VariableScaling(domain_length_scale=1.0)
        obj.add_conversion("natural", "SI", 2.0)
        obj.add_conversion("SI", "scaled", 2.0)

        # test that the 3 nodes are present
        assert {"natural", "SI", "scaled"} == set(obj.conversion_factors.nodes)
        # test that the edges are present (in both directions)
        assert len(obj.conversion_factors.edges) == 4

        assert obj.get_factor("natural", "SI") == 2.0
        assert obj.get_factor("SI", "natural") == 0.5
        assert obj.get_factor("SI", "scaled") == 2.0
        assert obj.get_factor("scaled", "SI") == 0.5
        assert obj.get_factor("natural", "scaled") == 4.0
        assert obj.get_factor("scaled", "natural") == 0.25

        assert obj.convert(1, "natural", "SI") == 2.0
        assert obj.convert(1, "SI", "natural") == 0.5
        assert obj.convert(1, "SI", "scaled") == 2.0
        assert obj.convert(1, "scaled", "SI") == 0.5
        assert obj.convert(1, "natural", "scaled") == 4.0
        assert obj.convert(1, "scaled", "natural") == 0.25


class TestEPScaling(unittest.TestCase):

    def test_ok(self):
        obj = scaling.EPScaling(domain_length_scale=1.0)
        # mm/day to m/s
        assert obj.convert(1.0, "natural", "SI") == 1e-3 / 24 / 60 / 60
        
        assert obj.convert(1.0, "natural", "scaled") == 1.0


class TestDistanceScaling(unittest.TestCase):

    def test_ok(self):
        obj = scaling.DistanceScaling(domain_length_scale=100.0)
        # m to m
        assert obj.convert(1.0, "natural", "SI") == 1.0

        # m to scaled
        assert obj.convert(1.0, "natural", "scaled") == 1 / 100
        assert obj.convert(1.0, "SI", "scaled") == 1 / 100


class TestWaterVaporFluxScaling(unittest.TestCase):

    def test_ok(self):
        obj = scaling.WaterVaporFluxScaling(domain_length_scale=22.25e5)
        # mb x m/s to m^2/s
        assert obj.convert(1.0, "natural", "SI") == 1.02e-2

        # mb x m/s to scaled
        self.assertAlmostEqual(
            obj.convert(1.0, "natural", "scaled"),
            1 / 2.525,
            places=3,
        )
