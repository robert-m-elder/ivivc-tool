# models/preprocessing.py

import numpy as np
import pandas as pd

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

def interpolate(t1,m1,t2,m2,N_interp=5):
    # (at least) N between subsequent points
    plist1 = []
    for i in range(len(m1)-1):
        plist1.extend(np.linspace(m1[i], m1[i+1], N_interp)[:-1])
    plist2 = []
    for i in range(len(m2)-1):
        plist2.extend(np.linspace(m2[i], m2[i+1], N_interp)[:-1])
    mm = np.array(sorted(set(plist1+plist2)))[::-1]
    plist1 = []
    for i in range(len(t1)-1):
        plist1.extend(np.linspace(t1[i], t1[i+1], N_interp)[:-1])
    plist2 = []
    for i in range(len(t2)-1):
        plist2.extend(np.linspace(t2[i], t2[i+1], N_interp)[:-1])
    tt = np.array(sorted(set(plist1+plist2)))
    # linear interpolation, with points beyond interpolation range set to nan
    ti1 = np.interp(mm,m1[::-1],t1[::-1], left=np.nan, right=np.nan) #[::-1]
    ti2 = np.interp(mm,m2[::-1],t2[::-1], left=np.nan, right=np.nan) #[::-1]
    mi1 = np.interp(tt,t1,m1, left=np.nan, right=np.nan)
    mi2 = np.interp(tt,t2,m2, left=np.nan, right=np.nan)
    return t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2

def interpolate_on_union(t1, m1, t2, m2):
    """
    Simplified version: interpolate only onto the exact union of original time points

    Parameters:
    -----------
    t1, m1 : array-like
    Dataset 1 (time, values)
    t2, m2 : array-like
    Dataset 2 (time, values)

    Returns:
    --------
    tuple
    (t1, m1, t2, m2, mm, tt, ti1, ti2, mi1, mi2)
    where tt contains only the union of original time points
    """

    # Convert to numpy arrays
    t1, m1 = np.array(t1), np.array(m1)
    t2, m2 = np.array(t2), np.array(m2)

    # Create union of original time points only
    time_union = set(t1).union(set(t2))
    tt = np.array(sorted(time_union))

    # Create union of original value points only
    value_union = set(m1).union(set(m2))
    mm = np.array(sorted(value_union, reverse=True)) # Descending order like your original

    # Interpolate onto the union grids
    mi1 = np.interp(tt, t1, m1, left=np.nan, right=np.nan)
    mi2 = np.interp(tt, t2, m2, left=np.nan, right=np.nan)

    ti1 = np.interp(mm, m1[::-1], t1[::-1], left=np.nan, right=np.nan)
    ti2 = np.interp(mm, m2[::-1], t2[::-1], left=np.nan, right=np.nan)

    return t1, m1, t2, m2, mm, tt, ti1, ti2, mi1, mi2


def no_interpolation(t1, m1, t2, m2):
    """Return analysis arrays without interpolating between observations.

    Values are aligned only at exact observed time/value matches. Missing
    positions are represented as NaN so downstream masks use only available
    raw observations.
    """
    t1, m1 = np.asarray(t1), np.asarray(m1)
    t2, m2 = np.asarray(t2), np.asarray(m2)

    def exact_values_on_grid(x, y, grid):
        out = np.full(len(grid), np.nan, dtype=float)
        for i, value in enumerate(grid):
            mask = x == value
            if np.any(mask):
                out[i] = np.nanmean(y[mask])
        return out

    tt = np.array(sorted(set(t1).union(set(t2))))
    mm = np.array(sorted(set(m1).union(set(m2)), reverse=True))

    mi1 = exact_values_on_grid(t1, m1, tt)
    mi2 = exact_values_on_grid(t2, m2, tt)
    ti1 = exact_values_on_grid(m1, t1, mm)
    ti2 = exact_values_on_grid(m2, t2, mm)

    return t1, m1, t2, m2, mm, tt, ti1, ti2, mi1, mi2


interpolations = {
    'none': no_interpolation,
    'default': interpolate_on_union,
    'alternative': interpolate,
}

