# tests/conftest.py

import numpy as np
import pytest

from utils.geometry import (
    load_airfoil_xy,
    rearrange_airfoil,
    resample_airfoil_cosine_finite_TE,
)

from utils.panel import data_preparation

@pytest.fixture
def airfoilcoords_path():
    return "data/NACA0012.txt"

@pytest.fixture
def raw_coords(airfoilcoords_path):
    return load_airfoil_xy(airfoilcoords_path)

""" 
Class: AirfoilSection
"""
@pytest.fixture
def geom(raw_coords):
    x, y = raw_coords
    xdata, ydata, chord, *_ = rearrange_airfoil(x, y)
    xnew, ynew = resample_airfoil_cosine_finite_TE(xdata, ydata, n_points=100)

    g = data_preparation(xnew, ynew, len(xnew))
    g["x"], g["y"] = xnew, ynew
    g["n_panels"] = len(g["xmid"])
    g["chord"] = chord
    g["beta"] = 0.0

    return g

"""
Class: FlowSection
"""
@pytest.fixture
def flow():
    a = np.deg2rad(5.0) # radians
    W = 50.0
    return {
        "U": float(W * np.cos(a)), # m/s
        "V": float(W * np.sin(a)), # m/s
        "W": float(W),
        "a_1": float(a),
        "rho": float(1.225),
        "Re": float(1.225 * W / 1.81e-5)
    }
