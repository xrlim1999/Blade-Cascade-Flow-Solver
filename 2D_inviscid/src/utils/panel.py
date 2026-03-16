import numpy as np
import math

def data_preparation(x, y, n_datapoints):

    ds     = np.zeros(n_datapoints-1)
    sine   = np.zeros(n_datapoints-1)
    cosine = np.zeros(n_datapoints-1)
    slope  = np.zeros(n_datapoints-1)
    xmid   = np.zeros(n_datapoints-1)
    ymid   = np.zeros(n_datapoints-1)

    # set initial x and y values
    x1 = x[0]
    y1 = y[0]

    # constant for tangent angle limits
    ex = 1e-6

    for n in range(0, n_datapoints-1):

        if n < n_datapoints-1:
            x2 = x[n+1]
            y2 = y[n+1]
        else:
            x2 = x[0]
            y2 = y[0]

        ds[n]     = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        sine[n]   = (y2 - y1) / ds[n]
        cosine[n] = (x2 - x1) / ds[n]

        abscos = abs(cosine[n])

        if abscos > ex:
            t = math.atan(sine[n]/cosine[n]) # compute angle of ds wrt horizontal axis
        else:
            t = None # as the division will blow up
        
        if abscos <= ex:
            slope[n] = (sine[n] / abs(sine[n])) * np.pi / 2.0 # sets slope to + or - (90 deg)
        elif cosine[n] > ex:
            slope[n] = t # angle within TANGENT quadrant
        elif cosine[n] < -ex:
            slope[n] = t - np.pi # angle within SINE quadrant

        # compute coordinates of pivotal points
        xmid[n] = (x1 + x2) * 0.5
        ymid[n] = (y1 + y2) * 0.5

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
