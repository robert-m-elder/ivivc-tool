# models/preprocessing.py

import numpy as np
import pandas as pd

def interpolate(t1,m1,t2,m2,N_interp=10):
    ## set zero to small value to avoid division errors
    #t1[t1==0] = 1e-2; t2[t2==0] = 1e-2
    # remove missing values due to unequal number of points in spreadsheet
    #mask = ~pd.isna(m2); t2,m2 = t2[mask], m2[mask]
    #mask = ~pd.isna(m1); t1,m1 = t1[mask], m1[mask]
    ## interpolate values onto same timepoints
    # constant N for whole range
    #tt = np.linspace(np.min(np.concatenate([t1,t2])), np.max(np.concatenate([t1,t2])), N_interp) # t-basis
    #mm = np.linspace(np.max(np.concatenate([m1,m2])), np.min(np.concatenate([m1,m2])), N_interp) # m-basis
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

interpolations = {
    'default_interpolation': interpolate,
}

