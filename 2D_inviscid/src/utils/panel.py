import numpy as np
import math

def data_preparation(x, y):
    """
    Compute panel geometry from a set of ordered airfoil nodes.

    For each panel connecting adjacent nodes, computes the panel length,
    sine and cosine of the panel tangent angle, the tangent slope angle,
    and the panel midpoint coordinates. The final panel connects the last
    node back to the first, closing the loop.

    Parameters
    ----------
    x, y : array-like
        Ordered airfoil node coordinates, shape (n,).
        Expected ordering: TE->upper->LE->lower->TE.

    Returns
    -------
    geom : dict
        Dictionary containing panel geometry arrays, each of shape (n,):
        
        "ds"     : panel lengths [m]
        "sine"   : sine of panel tangent angle
        "cosine" : cosine of panel tangent angle
        "slope"  : panel tangent angle measured from horizontal [rad]
        "xmid"   : x coordinates of panel midpoints
        "ymid"   : y coordinates of panel midpoints
    """
    n = len(x) # number of datatpoints

    ds     = np.zeros(n)
    sine   = np.zeros(n)
    cosine = np.zeros(n)
    slope  = np.zeros(n)
    xmid   = np.zeros(n)
    ymid   = np.zeros(n)

    # set initial x and y values
    x1 = x[0]
    y1 = y[0]

    # constant for tangent angle limits
    ex = 1e-6

    for k in range(0, n):

        if k < n-1:
            x2, y2 = x[k+1], y[k+1]
        else:
            x2, y2 = x[0], y[0]

        ds[k]     = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        sine[k]   = (y2 - y1) / ds[k]
        cosine[k] = (x2 - x1) / ds[k]

        abscos = abs(cosine[k])

        if abscos > ex:
            t = math.atan(sine[k]/cosine[k]) # compute angle of ds wrt horizontal axis
        else:
            t = None # as the division will blow up
        
        if abscos <= ex:
            slope[k] = (sine[k] / abs(sine[k])) * np.pi / 2.0 # sets slope to + or - (90 deg)
        elif cosine[k] > ex:
            slope[k] = t # angle within TANGENT quadrant
        elif cosine[k] < -ex:
            slope[k] = t - np.pi # angle within SINE quadrant

        # compute coordinates of pivotal points
        xmid[k] = (x1 + x2) * 0.5
        ymid[k] = (y1 + y2) * 0.5

        # move to next coordinates
        x1, y1 = x2, y2
    
    geom = dict()
    geom["ds"    ] = ds
    geom["sine"  ] = sine
    geom["cosine"] = cosine
    geom["slope" ] = slope
    geom["xmid"  ] = xmid
    geom["ymid"  ] = ymid

    return geom
