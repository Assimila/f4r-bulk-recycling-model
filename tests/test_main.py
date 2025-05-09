import unittest

import numpy as np

from bulk_recycling_model import preprocess
from bulk_recycling_model.main import run
from bulk_recycling_model.scaling import Scaling, UnitSystem
from tests.data.load_data import load_data


class Test_run(unittest.TestCase):

    def setUp(self):
        dat = load_data()
        lon = dat["lon"]
        lat = dat["lat"]

        L = lon[-1] - lon[0]  # degrees
        # convert to meters
        L = L * 111e3 * np.cos(np.deg2rad(lat.mean()))

        dx = L / (len(lon) - 1)  # meters

        H = lat[-1] - lat[0]  # degrees
        # convert to meters
        H = H * 111e3

        dy = H / (len(lat) - 1)  # meters

        # get a scaling object to convert from natural to scaled units
        scaling = Scaling(H)

        dx = scaling.distance.convert(dx, UnitSystem.SI, UnitSystem.scaled)
        dy = scaling.distance.convert(dy, UnitSystem.SI, UnitSystem.scaled)

        Fx = dat["Fx"]
        Fx = scaling.water_vapor_flux.convert(Fx, UnitSystem.natural, UnitSystem.scaled)
        Fx_left = preprocess.prepare_Fx_left(Fx)
        Fx_right = preprocess.prepare_Fx_right(Fx)

        Fy = dat["Fy"]
        Fy = scaling.water_vapor_flux.convert(Fy, UnitSystem.natural, UnitSystem.scaled)
        Fy_bottom = preprocess.prepare_Fy_bottom(Fy)
        Fy_top = preprocess.prepare_Fy_top(Fy)

        E = dat["E"]
        E = scaling.evaporation.convert(E, UnitSystem.natural, UnitSystem.scaled)
        E = preprocess.prepare_E(E)

        P = preprocess.calculate_precipitation(Fx_left, Fx_right, Fy_bottom, Fy_top, E, dx, dy)

        self.Fx_left = Fx_left
        self.Fx_right = Fx_right
        self.Fy_bottom = Fy_bottom
        self.Fy_top = Fy_top
        self.E = E
        self.P = P
        self.dx = dx
        self.dy = dy

    def test_ok(self):
        """
        Can run the model with the test data
        """
        status = run(
            self.Fx_left,
            self.Fx_right,
            self.Fy_bottom,
            self.Fy_top,
            self.E,
            self.P,
            self.dx,
            self.dy,
            tol=1e-2,
        )
        assert status["success"]
        assert status["k"] > 0
        assert len(status["deltas"]) == status["k"]
