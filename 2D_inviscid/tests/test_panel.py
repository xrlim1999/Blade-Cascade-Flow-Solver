"""
tests/test_panel.py
----------------------
Unit tests for utils/panel.py

Functions tested:
    - data_preparation
"""

import numpy as np
import pytest

from utils.geometry import (
    rearrange_airfoil,
    resample_airfoil_cosine_finite_TE,
    rotate_airfoil_about_te
)
from utils.panel import data_preparation

class TestDataPreparation:

    @pytest.fixture(autouse=True)
    def setup(self, raw_coords):
        """ Runs once per class — stores [cache] coords on self for all tests to use """
        self.x, self.y = raw_coords

        self.xdata, self.ydata, *_ = rearrange_airfoil(self.x, self.y)

        self.n_points = 100

        self.xnew, self.ynew = resample_airfoil_cosine_finite_TE(
            self.xdata, self.ydata, n_points=self.n_points
        )

        self.beta_deg = 10.0
        self.beta_rad = np.deg2rad(self.beta_deg)
        self.x_rotated, self.y_rotated = rotate_airfoil_about_te(self.xnew, self.ynew, self.beta_rad)

        self.geom = data_preparation(self.x_rotated, self.y_rotated)
    
    def test_number_of_panels(self):
        assert len(self.geom["xmid"]) == len(self.geom["ymid"])
        assert len(self.geom["xmid"]) == self.n_points + 1
    