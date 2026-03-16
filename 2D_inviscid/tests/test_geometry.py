"""
tests/test_geometry.py
----------------------
Unit tests for utils/geometry.py

Functions tested:
    - load_airfoil_xy
    - rearrange_airfoil
    - resample_airfoil_cosine_finite_TE
    - rotate_airfoil_about_te
"""

import numpy as np
import pytest

from utils.geometry import (
    load_airfoil_xy,
    rearrange_airfoil,
    resample_airfoil_cosine_finite_TE,
    rotate_airfoil_about_te
)

# ===========================================================================
# Test Case 1: load_airfoil_xy
# ===========================================================================
class TestLoadAirfoilXY:

    @pytest.fixture(autouse=True)
    def coords_setup(self, raw_coords):
        """ Runs once per class — stores [cache] coords on self for all tests to use """
        self.x, self.y = raw_coords

    def test_returns_two_arrays(self):
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)

    def test_arrays_are_float(self):
        assert self.x.dtype == float
        assert self.y.dtype == float

    def test_arrays_are_same_length(self):
        assert len(self.x) == len(self.y)

    def test_raises_on_invalid_file(self):
        with pytest.raises(Exception):
            load_airfoil_xy("data/invalidfile.txt")
    
# ===========================================================================
# Test Case 2: rearrange_airfoil
# ===========================================================================
class TestRearrangeAirfoil:

    @pytest.fixture(autouse=True)
    def coords_setup(self, raw_coords):
        """ Runs once per class — stores [cache] coords on self for all tests to use """
        self.x, self.y = raw_coords
        (
            self.xdata,
            self.ydata,
            self.chord,
            self.x_le,
            self.y_le,
            self.x_te,
            self.y_te,
            self.max_thick
        ) = rearrange_airfoil(self.x, self.y)

    def test_chord_is_normalised(self):
        assert self.chord == 1.0

    def test_x_range_normalised(self):
        """ x-coords must lie within [0, 1] """
        assert self.xdata.min() >= -1e-10
        assert self.xdata.max() <= 1.0 + 1e-10

    def test_le_and_te_are_correct(self):
        assert self.x_le == pytest.approx(self.xdata.min())
        assert self.x_te == pytest.approx(self.xdata.max())
        assert self.y_le == pytest.approx(self.ydata[np.argmin(self.xdata)])

        # --- average TE coordinates ---
        te_inds = np.where(np.isclose(self.xdata, np.max(self.xdata), atol=1e-8, rtol=0.0))[0]
        if te_inds.size == 2:
            y_te_expected = 0.5 * (self.ydata[te_inds[0]] + self.ydata[te_inds[1]])
        else:
            y_te_expected = self.ydata[te_inds[0]]

        assert self.y_te == pytest.approx(y_te_expected)

    def test_le_near_zero(self):
        """ Leading Edge should be very near or at 0 """
        assert self.x_le == pytest.approx(0.0, abs=1e-6)

    def test_te_near_one(self):
        """ Trailing Edge should be very near or at 1.0 """
        assert self.x_te == pytest.approx(1.0, abs=1e-6)

    def test_new_arrays_are_same_length(self):
        assert len(self.xdata) == len(self.ydata)

    def test_max_thickness(self):
        assert self.max_thick > 0.0
        assert self.max_thick == pytest.approx(0.12, rel=0.01)

# ===========================================================================
# Test Case 3: resample_airfoil_cosine_finite_TE
# ===========================================================================
class TestResampleAirfoilCosineFiniteTE:

    @pytest.fixture(autouse=True)
    def setup(self, raw_coords):
        """ Runs once per class — stores [cache] coords on self for all tests to use """
        self.x, self.y = raw_coords

        self.xdata, self.ydata, *_ = rearrange_airfoil(self.x, self.y)

        self.n_points = 100

        self.xnew, self.ynew = resample_airfoil_cosine_finite_TE(
            self.xdata, self.ydata, n_points=self.n_points
        )
    
    def test_array_returns_two_arrays(self):
        assert isinstance(self.xnew, np.ndarray)
        assert isinstance(self.ynew, np.ndarray)
    
    def test_array_lengths_are_same(self):
        assert len(self.xnew) == len(self.ynew)

    def test_open_loop(self):
        assert self.ynew[0] != pytest.approx(self.ynew[-1], abs=1e-6)
    
    def test_single_le_point(self):
        assert np.where(np.isclose(self.xnew, np.min(self.xnew), atol=1e-8, rtol=0.0))[0].size == 1
    
    def test_odd_n_points_rounded_up_to_even(self):
        xnew, _ = resample_airfoil_cosine_finite_TE(self.xdata, self.ydata, n_points=99)
        assert len(xnew) == 101

    def test_x_stays_within_chord(self):
        assert self.xnew.min() >= -1e-10
        assert self.xnew.max() <= 1.0 + 1e-10

    def test_le_exists_near_zero(self):
        assert self.xnew.min() == pytest.approx(0.0, abs=1e-6)

    def test_te_exists_near_one(self):
        assert self.xnew.max() == pytest.approx(1.0, abs=1e-6)

    def test_cosine_clustering_near_le_and_te(self):
        gaps = np.abs(np.diff(self.xnew))
        mid = len(gaps) // 4
        assert gaps[mid] > gaps[0]
        assert gaps[len(self.xnew)-mid] > gaps[-1]

# ===========================================================================
# Test Case 4: rotate_airfoil_about_te
# ===========================================================================
class TestRotateAirfoilAboutTe:

    @pytest.fixture
    def setup(self, raw_coords):
        """ Runs once per class — stores [cache] coords on self for all tests to use """
        self.x, self.y = raw_coords

        self.xdata, self.ydata, *_ = rearrange_airfoil(self.x, self.y)

        self.n_points = 100

        xnew, ynew = resample_airfoil_cosine_finite_TE(
            self.xdata, self.ydata, n_points=self.n_points
        )

        self.beta_deg = 10.0
        self.beta_rad = np.deg2rad(self.beta_deg)
        self.xnew, self.ynew = rotate_airfoil_about_te(xnew, ynew, self.beta_rad, pivot="mid")

    