# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Import modules
import os
import sys

import xarray as xr
import numpy as np
import pandas as pd
import numpy.ma as ma
import scipy.stats as stats
import scipy.signal as signal
from scipy.signal import savgol_filter
from scipy.stats import linregress

# =============================================================================
# least square linear trend
def linear_trend_detrend(y):
    """
    Detrend the y by subtracting the linear fit.
    
    Parameters:
    - y: Array-like y values to be detrended
    - x: Corresponding x-values (e.g., time or years)
    
    Returns:
    - detrended_data: Data after subtracting the linear fit
    """
    x = np.arange(len(y))
    coeff = np.polyfit(x, y, 1)
    linear_fit = np.polyval(coeff, x)
    return y - linear_fit

# =============================================================================
# Separate the time series into segments of 65 years with 20 years overlap
def generate_segments(total_years=1000, segment_length=65, overlap=20):
    # Generate a time series data for total_years
    time_series_data = np.arange(total_years)
    
    # Initialize a list to store the segments
    segments = []
    
    # Create segments
    start_year = 0
    while start_year + segment_length <= total_years:
        end_year = start_year + segment_length
        segment = time_series_data[start_year:end_year]
        segments.append(segment)
        start_year = start_year + segment_length - overlap
        
    return segments

# Example usage
segments = generate_segments()

# Display the first 3 segments as an example
for i, segment in enumerate(segments[:3]):
    print(f"Segment {i+1}: {segment[0]}-{segment[-1]}")