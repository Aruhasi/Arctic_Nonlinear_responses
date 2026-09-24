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
import pymannkendall as mk

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy.ma as ma
# preprocess the data to remove the nan values
def preprocess_nan(data):
    """
    preprocess the data to remove the nan values
    """
    data = np.nan_to_num(data)
    return data 

# def preprocess_nan(data, fill_value=0.0):
#     """
#     Replace NaN values in the data with a specified value.
#     """
#     data = np.nan_to_num(data, nan=fill_value)
#     return data
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# np.roll apply to the longitude dimension with 0-360 degree
def convert_longitude(data):
    """
    Convert longitudes in an xarray DataArray or Dataset from 0–360 to -180–180.

    Parameters:
    - data: xarray.DataArray or xarray.Dataset with 'lon' coordinate.

    Returns:
    - data: Same object with adjusted 'lon' coordinate.
    """
    data = data.copy()
    data['lon'] = ((data['lon'] + 180) % 360) - 180
    return data.sortby('lon')
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# np.roll apply to the longitude dimension with 0-360 degree
def adjust_longitude(data, original_lons):
    """
    Adjust the data array and corresponding longitudes from a 0-360 to a -180 to 180 range.
    
    Parameters:
    - data: 2D numpy array with shape (latitude, longitude)
    - original_lons: 1D numpy array of longitudes from 0 to 360
    
    Returns:
    - data_rolled: The data array adjusted so longitudes range from -180 to 180
    - lons_adjusted: The adjusted longitude array from -180 to 180
    """
    # Calculate the index where we need to split and roll the array
    shift_index = len(original_lons) // 2

    # Roll the data array to shift the longitudes
    data_rolled = np.roll(data, shift=shift_index, axis=1)

    # Roll the longitude array and adjust values to be within -180 to 180
    lons_rolled = np.roll(original_lons, shift=shift_index)
    lons_rolled[lons_rolled >= 180] -= 360

    return data_rolled, lons_rolled
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from scipy.signal.windows import lanczos
"""
References
----------

    Duchon C. E. (1979) Lanczos Filtering in One and Two Dimensions.
    Journal of Applied Meteorology, Vol 18, pp 1016-1022.

"""
def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def lanczos_filter(window_size, cutoff):
    """
    Create a Lanczos filter.

    :param size: The number of points in the filter.
    :param a: The Lanczos parameter, which determines the size of the window.
    :return: A 1D numpy array containing the Lanczos filter.
    """
    if window_size % 2 == 0:
        raise ValueError("window_size of the filter must be odd")

    def lanczos_window(x):
        return np.sinc(x) * np.sinc(x / cutoff)

    n = (window_size - 1) // 2
    x = np.linspace(-n, n, window_size)
    filter = lanczos_window(x)

    return filter / np.sum(filter)  # Normalize the filter

def apply_lanczos_filter(data, window, cutoff):
    filter = lanczos_filter(window, cutoff)
    filtered_data = np.convolve(data, filter, mode='same')
    return filtered_data

def apply_lanczos_filter_3d(data, window, cutoff):
    # Assuming data is a 3D numpy array with dimensions [time, lat, lon]
    filtered_data = np.empty_like(data)

    # Iterate over each spatial point
    for lat in range(data.shape[1]):
        for lon in range(data.shape[2]):
            # Extract the time series for this point
            time_series = data[:, lat, lon]

            # Apply the 1D filter to this time series
            filtered_time_series = np.convolve(time_series, lanczos_filter(window,cutoff), mode='same')

            # Store the filtered time series back into the 3D array
            filtered_data[:, lat, lon] = filtered_time_series

    return filtered_data
