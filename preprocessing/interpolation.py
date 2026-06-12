# models/preprocessing.py

import numpy as np
import pandas as pd

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

#def interpolate_to_common_mass_grid(t1, m1, t2, m2, num_points=100, mass_range=None, method='linear'):
#    """
#    Interpolate two time-mass datasets onto a common mass grid.
#    
#    Parameters:
#    -----------
#    t1, m1 : array-like
#        Time and mass arrays for dataset 1
#    t2, m2 : array-like  
#        Time and mass arrays for dataset 2
#    num_points : int
#        Number of points in the common mass grid
#    mass_range : tuple or None
#        (min_mass, max_mass) for interpolation range. If None, uses overlap of both datasets
#    method : str
#        Interpolation method: 'linear', 'nearest', or 'cubic'
#    
#    Returns:
#    --------
#    'mass_grid': Common mass grid
#    't1_interp': Interpolated time values for dataset 1
#    't2_interp': Interpolated time values for dataset 2
#    """
#    
#    # Convert to numpy arrays
#    t1, m1 = np.asarray(t1), np.asarray(m1)
#    t2, m2 = np.asarray(t2), np.asarray(m2)
#    
#    # Determine mass range for interpolation
#    if mass_range is None:
#        min_mass = max(np.min(m1), np.min(m2))
#        max_mass = min(np.max(m1), np.max(m2))
#    else:
#        min_mass, max_mass = mass_range
#    
#    # Create common mass grid
#    mass_grid = np.linspace(min_mass, max_mass, num_points)
#
#    # Alternative approach: Sort by mass and use standard interpolation on monotonic segments
#    def interpolate_monotonic_segments(t_data, m_data, target_masses, interp_method):
#        """
#        Split data into monotonic segments and interpolate each segment
#        """
#        interpolated_times = np.full(len(target_masses), np.nan)
#        
#        # Find monotonic segments
#        mass_diff = np.diff(m_data)
#        direction_changes = np.where(np.diff(np.sign(mass_diff)))[0] + 1
#        
#        # Add start and end points
#        segment_boundaries = np.concatenate([[0], direction_changes, [len(m_data)]])
#        
#        for i in range(len(segment_boundaries) - 1):
#            start_idx = segment_boundaries[i]
#            end_idx = segment_boundaries[i + 1]
#            
#            seg_t = t_data[start_idx:end_idx]
#            seg_m = m_data[start_idx:end_idx]
#            
#            # Skip if segment is too short
#            if len(seg_m) < 2:
#                continue
#            
#            # Sort by mass for this segment
#            sort_indices = np.argsort(seg_m)
#            seg_m_sorted = seg_m[sort_indices]
#            seg_t_sorted = seg_t[sort_indices]
#            
#            # Remove duplicates in mass
#            unique_indices = np.unique(seg_m_sorted, return_index=True)[1]
#            seg_m_unique = seg_m_sorted[unique_indices]
#            seg_t_unique = seg_t_sorted[unique_indices]
#            
#            if len(seg_m_unique) < 2:
#                continue
#            
#            # Find target masses within this segment's range
#            seg_min, seg_max = np.min(seg_m_unique), np.max(seg_m_unique)
#            valid_targets = (target_masses >= seg_min) & (target_masses <= seg_max)
#            
#            if np.any(valid_targets):
#                target_subset = target_masses[valid_targets]
#                
#                # Interpolate
#                if interp_method == 'nearest':
#                    interp_func = interp1d(seg_m_unique, seg_t_unique, kind='nearest', 
#                                         bounds_error=False, fill_value=np.nan)
#                elif interp_method == 'linear':
#                    interp_func = interp1d(seg_m_unique, seg_t_unique, kind='linear',
#                                         bounds_error=False, fill_value=np.nan)
#                elif interp_method == 'cubic' and len(seg_m_unique) >= 4:
#                    interp_func = interp1d(seg_m_unique, seg_t_unique, kind='cubic',
#                                         bounds_error=False, fill_value=np.nan)
#                else:
#                    interp_func = interp1d(seg_m_unique, seg_t_unique, kind='linear',
#                                         bounds_error=False, fill_value=np.nan)
#                
#                interp_values = interp_func(target_subset)
#                
#                # Update results (prefer earlier segments for overlapping ranges)
#                mask = valid_targets & np.isnan(interpolated_times)
#                interpolated_times[mask] = interp_values[mask[valid_targets]]
#        
#        return interpolated_times
#    
#    # Use the monotonic segments approach as it's more robust
#    t1_interp = interpolate_monotonic_segments(t1, m1, mass_grid, method)
#    t2_interp = interpolate_monotonic_segments(t2, m2, mass_grid, method)
#    
#    # Create validity mask
#    valid_mask = ~(np.isnan(t1_interp) | np.isnan(t2_interp))
#    
#    return mass_grid, t1_interp, t2_interp
#
#import numpy as np
#from scipy.interpolate import interp1d
#
#def interpolate_to_common_time_grid(t1, m1, t2, m2, point_density=100, interpolation_method='linear', extrapolation_mode='constant'):
#    """
#    Interpolate two time-series datasets onto a common time grid.
#
#    Parameters:
#    -----------
#    t1, m1 : array-like
#        Time and mass data for first dataset
#    t2, m2 : array-like
#        Time and mass data for second dataset
#    point_density : int, optional
#        Number of points in the interpolated time grid (default: 100)
#    interpolation_method : str, optional
#        Interpolation method ('linear', 'cubic', 'quadratic', etc.) (default: 'linear')
#    extrapolation_mode : str, optional
#        How to handle extrapolation beyond data bounds:
#        - 'constant': use boundary values
#        - 'extrapolate': extend the interpolation
#        - 'nan': return NaN outside bounds (default: 'constant')
#
#    Returns:
#    --------
#    t_common : ndarray
#        Common time grid
#    m1_interp : ndarray
#        Interpolated mass values for dataset 1
#    m2_interp : ndarray
#        Interpolated mass values for dataset 2
#    """
#
#    # Convert inputs to numpy arrays
#    t1, m1 = np.asarray(t1), np.asarray(m1)
#    t2, m2 = np.asarray(t2), np.asarray(m2)
#
#    # Validate inputs
#    if len(t1) != len(m1) or len(t2) != len(m2):
#        raise ValueError("Time and mass arrays must have the same length")
#
#    if len(t1) < 2 or len(t2) < 2:
#        raise ValueError("Each dataset must have at least 2 points")
#
#    # Sort datasets by time (in case they're not already sorted)
#    sort_idx1 = np.argsort(t1)
#    sort_idx2 = np.argsort(t2)
#    t1_sorted, m1_sorted = t1[sort_idx1], m1[sort_idx1]
#    t2_sorted, m2_sorted = t2[sort_idx2], m2[sort_idx2]
#
#    # Define common time range
#    t_min = max(np.min(t1_sorted), np.min(t2_sorted))
#    t_max = min(np.max(t1_sorted), np.max(t2_sorted))
#
#    if t_min >= t_max:
#        raise ValueError("Datasets have no overlapping time range")
#
#    # Create common time grid
#    t_common = np.linspace(t_min, t_max, point_density)
#
#    # Set up extrapolation behavior
#    fill_value = 'extrapolate' if extrapolation_mode == 'extrapolate' else None
#    bounds_error = extrapolation_mode == 'extrapolate'
#
#    if extrapolation_mode == 'constant':
#        fill_value = (m1_sorted[0], m1_sorted[-1])
#    elif extrapolation_mode == 'nan':
#        fill_value = np.nan
#
#    # Create interpolation functions
#    try:
#        interp_func1 = interp1d(t1_sorted, m1_sorted,
#                               kind=interpolation_method,
#                               bounds_error=bounds_error,
#                               fill_value=fill_value)
#
#        interp_func2 = interp1d(t2_sorted, m2_sorted,
#                               kind=interpolation_method,
#                               bounds_error=bounds_error,
#                               fill_value=fill_value)
#    except ValueError as e:
#        raise ValueError(f"Interpolation failed: {e}")
#
#    # Perform interpolation
#    m1_interp = interp_func1(t_common)
#    m2_interp = interp_func2(t_common)
#
#    return t_common, m1_interp, m2_interp
#
#def interpolate(t1,m1,t2,m2,N_interp=20):
#    tt, mi1, mi2 = interpolate_to_common_time_grid(t1, m1, t2, m2, point_density=N_interp, interpolation_method='linear', extrapolation_mode='constant')
#    mm, ti1, ti2 = interpolate_to_common_mass_grid(t1, m1, t2, m2, num_points=N_interp, mass_range=None, method='linear')
#    return t1,m1,t2,m2,mm,tt,ti1,ti2,mi1,mi2

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


interpolations = {
    'default': interpolate_on_union,
    'alternative': interpolate,
}

