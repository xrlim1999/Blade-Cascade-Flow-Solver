from __future__ import annotations
import numpy as np
from pathlib import Path

def load_airfoil_xy(path: str | Path) -> tuple[np.ndarray, np.ndarray]:

    """Load aerofoil coordinates from a two-column text file (x, y)."""
    data = np.loadtxt(path)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected 2+ columns, got shape {data.shape}")
    
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    
    return x, y